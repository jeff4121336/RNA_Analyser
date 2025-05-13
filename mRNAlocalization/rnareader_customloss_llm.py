# import RNA-FM LLM
# define the classifer layer (7 binary problems with OvR)
import torch
import numpy as np
import fm
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.optim as optim
import os
import matplotlib.pyplot as plt

from gene_data import Genedata
from hier_attention_mask import AttentionMask
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./data/"
DATA_FILE = DATA_PATH + "modified_multilabel_seq_nonredundent.fasta"
#DATA_FILE = DATA_PATH + "test.fasta"

EPOCHS = 100
# Update the THRESHOLD global variable with class-specific thresholds

THRESHOLD = [0.59, 0.98, 0.16, 0.52, 0.30, 0.25, 0.12] #CNN LINEAR MODIFIED LOSS
#THRESHOLD = [0.59, 0.99, 0.14, 0.50, 0.24, 0.22, 0.08] #CNN MEAN MODIFIED LOSS

encoding_seq = OrderedDict([
	('A', [1, 0, 0, 0]),
	('C', [0, 1, 0, 0]),
	('G', [0, 0, 1, 0]),
	('T', [0, 0, 0, 1]),
	('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
	('-', [0, 0, 0, 0]),  # Pad
])

class StableFocalLoss(torch.nn.Module):
	def __init__(self, alpha, gamma, omega, reduction='sum'):
		super().__init__()
		self.alpha = alpha  # Tensor of shape [N_classes]
		self.gamma = gamma  # Scalar
		self.omega = omega  # Tensor of shape [N_classes]
		self.reduction = reduction

	def forward(self, logits, targets):
		# Logit-adjusted focal loss (numerically stable)
		bce_loss = F.binary_cross_entropy_with_logits(
			logits, targets, reduction='none'
		)  # Shape: [batch_size, N_classes]

		probs = torch.sigmoid(logits)
		p_t = torch.where(targets == 1, probs, 1 - probs)
		modulating_factor = (1 - p_t) ** self.gamma

		# Apply class weights (alpha) and category weights (omega)
		loss = modulating_factor * bce_loss
		loss = loss * self.alpha * self.omega  # Element-wise multiplication

		if self.reduction == 'sum':
			return loss.sum()
		elif self.reduction == 'mean':
			return loss.mean()
		else:
			return loss

class EarlyStopper:
	def __init__(self, patience=3, min_delta=0.05):
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.min_validation_loss = float('inf')

	def early_stop(self, validation_loss):
		#print(f'Validation Loss: {validation_loss:.5f}, Min Validation Loss: {self.min_validation_loss:.5f}, Counter: {self.counter}')
		if validation_loss < self.min_validation_loss:
			self.min_validation_loss = validation_loss
			self.counter = 0
		elif validation_loss > (self.min_validation_loss + self.min_delta):
			self.counter += 1
			if self.counter >= self.patience:
				return True
		return False

class RNA_FM:
	def __init__(self):
		embedding_model, alphabet = fm.pretrained.rna_fm_t12()
		batch_converter = alphabet.get_batch_converter()
		embedding_model.eval()  # disables dropout for deterministic results
		self.model = embedding_model
		self.batch_converter = batch_converter
		self.max_length = 1000  # Maximum sequence length for rna_fm

	def embeddings(self, data):
		"""
		Generate embeddings for sequences of length 8000, handling sequences longer than max_length.
		The embeddings are generated per nucleotide, and chunks are simply concatenated.
		"""
		all_embeddings = []
		self.model = self.model.to(DEVICE)

		for label, sequence in data:
			chunks = [
				sequence[i:i + self.max_length]
				for i in range(0, len(sequence), self.max_length)
			]
			chunk_embeddings = []
			for chunk in chunks:
				batch_data = [(label, chunk)]
				batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_data)
				batch_tokens = batch_tokens.to(DEVICE)

				with torch.no_grad():
					results = self.model(batch_tokens, repr_layers=[12])
				chunk_embedding = results["representations"][12]  # Shape: [1, seq_len, 640]
				chunk_embeddings.append(chunk_embedding)

			combined_embedding = torch.cat(chunk_embeddings, dim=1)  # Shape: [1, seq_len, 640]
			combined_embedding = torch.mean(combined_embedding, dim=1)  # Shape: [1, 640]
			all_embeddings.append(combined_embedding)
			
		# Concatenate all embeddings along the batch dimension
		return torch.cat(all_embeddings, dim=0)  # Shape: [batch_size, seq_len, 640]

class MultiscaleCNNLayers(nn.Module):
	def __init__(self, in_channels, embedding_dim, pooling_size, pooling_stride, drop_rate_cnn, drop_rate_fc, length, nb_classes):
		super(MultiscaleCNNLayers, self).__init__()

		self.bn1 = nn.BatchNorm1d(in_channels)
		self.bn2 = nn.BatchNorm1d(in_channels // 2)

		self.conv1_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=in_channels, kernel_size=9, padding=4)
		self.conv1_2 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=9, padding=4)
		self.conv2_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=in_channels, kernel_size=20, padding=10)
		self.conv2_2 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=20, padding=10)
		self.conv3_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=in_channels // 2, kernel_size=49, padding=24)
		self.conv3_2 = nn.Conv1d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=49, padding=24)

		self.pool = nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_stride)

		self.dropout_cnn = nn.Dropout(drop_rate_cnn)
		self.dropout_fc = nn.Dropout(drop_rate_fc)

		self.fc = nn.Linear(length // pooling_stride, 7)

		nn.init.xavier_uniform_(self.fc.weight)


		self.activation = nn.GELU() 

		self.attentionHead = AttentionMask(hidden = (length // pooling_stride) + 1, da = 80, r = 5)

	def forward_cnn(self, x, conv1, conv2, bn1, bn2):
		x = conv1(x)
		x = self.activation(bn1(x))
		x = conv2(x)
		x = self.activation(bn2(x))
		x = self.pool(x)
		x, regulatization_loss, att_map = self.attentionHead(x)
		x = self.dropout_cnn(x)
		return x, regulatization_loss
	
class MultiscaleCNNModel(nn.Module):
	def __init__(self, layers, num_classes, sequence_length, aggregation_method="learnable_linear"):
		super(MultiscaleCNNModel, self).__init__()
		self.layers = layers
		self.aggregation_method = aggregation_method 
		
		if aggregation_method == "learnable_linear":
			self.sequence_aggregator = nn.Linear(sequence_length, 1)

	def forward(self, x):
		x1, x1_regulatization_loss = self.layers.forward_cnn(x, self.layers.conv1_1, self.layers.conv1_2, self.layers.bn1, self.layers.bn2)
		x2, x2_regulatization_loss = self.layers.forward_cnn(x, self.layers.conv2_1, self.layers.conv2_2, self.layers.bn1, self.layers.bn2)
		x3, x3_regulatization_loss = self.layers.forward_cnn(x, self.layers.conv3_1, self.layers.conv3_2, self.layers.bn2, self.layers.bn2)
		x = torch.cat((x1, x2, x3), dim=1) # [32, 15, 999]
		x = self.layers.dropout_fc(self.layers.fc(x)) # [32, 15, 7]

		if self.aggregation_method == "learnable_linear":
			x = x.permute(0, 2, 1)  # Shape: [batch_size, 7, 15]
			x = self.sequence_aggregator(x).squeeze(-1) 

		return x, x1_regulatization_loss + x2_regulatization_loss + x3_regulatization_loss

class CombinedModel(nn.Module):
	def __init__(self, cnn_output_dim, llm_output_dim, hidden_dim, nb_classes):
		super(CombinedModel, self).__init__()
		self.nb_classes = nb_classes

		self.parallel_layers = nn.ModuleList([
			nn.Sequential(
				nn.Linear(2 + llm_output_dim, hidden_dim),
				nn.ReLU(),
				nn.Dropout(0.2),
				nn.Linear(hidden_dim, 1)  # Output 1 label per layer
			)
			for _ in range(nb_classes)
		])

	def forward(self, cnn_output, llm_output):
		# cnn - [batch_size, cnn_output_dim] / llm - [batch_size, 1, llm_output_dim]
		llm_output = llm_output.squeeze(1)  # Remove the second dimension if present

		
		outputs = []
		for i, layer in enumerate(self.parallel_layers):
			specific_label = cnn_output[:, i].unsqueeze(1)  # Shape: [batch_size, 1]
			last_label = cnn_output[:, -1].unsqueeze(1)  # Shape: [batch_size, 1]

			x = torch.cat((specific_label, last_label, llm_output), dim=1)  # Shape: [batch_size, 2 + llm_output_dim]

			outputs.append(layer(x))  # Shape: [batch_size, 1]

		# Concatenate outputs from all parallel layers
		outputs = torch.cat(outputs, dim=1)  # Shape: [batch_size, nb_classes]

		return outputs
	
class GeneDataset(Dataset):
	def __init__(self, data, labels):
		self.data = data
		self.labels = labels

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		combined_sequence, rnafm_embedding, one_hot_encoding = self.data[idx]
		return combined_sequence, rnafm_embedding, one_hot_encoding, self.labels[idx]
	
def get_id_label_seq_Dict(gene_data):
#	{
#   	gene_id_1: {label_1: (seqLeft_1, seqRight_1)},
#    	gene_id_2: {label_2: (seqLeft_2, seqRight_2)}, ...
#	}
	id_label_seq_Dict = OrderedDict()
	for gene in gene_data:
		label = gene.label
		gene_id = gene.id.strip()
		id_label_seq_Dict[gene_id] = {}
		id_label_seq_Dict[gene_id][label]= (gene.seqLeft, gene.seqRight)

	return id_label_seq_Dict

def get_label_id_Dict(id_label_seq_Dict):
#	{
#   	label_1: {gene_id_1, gene_id_2, ...},
#    	label_2: {gene_id_3, gene_id_4}, ...
#	}
	label_id_Dict = OrderedDict()
	for eachkey in id_label_seq_Dict.keys():
		label = list(id_label_seq_Dict[eachkey].keys())[0]
		label_id_Dict.setdefault(label,set()).add(eachkey)
	
	return label_id_Dict

def group_sample(label_id_Dict,datasetfolder,foldnum=8):
	Train = OrderedDict()
	Test = OrderedDict()
	Val = OrderedDict()
	for i in range(foldnum):
		Train.setdefault(i,list())
		Test.setdefault(i,list())
		Val.setdefault(i,list())
	
	for eachkey in label_id_Dict:
		label_ids = list(label_id_Dict[eachkey])
		if len(label_ids)<foldnum:
			for i in range(foldnum):
				Train[i].extend(label_ids)
			continue
		
		[train_fold_ids, val_fold_ids,test_fold_ids] = KFoldSampling(label_ids, foldnum)
		for i in range(foldnum):
			Train[i].extend(train_fold_ids[i])
			Val[i].extend(val_fold_ids[i])
			Test[i].extend(test_fold_ids[i])
	
	for i in range(foldnum):
		print('Train length: %s, Test length: %s, Val length: %s'%(len(Train[i]),len(Test[i]),len(Val[i])))
		np.savetxt(datasetfolder+'/Train'+str(i)+'.txt', np.asarray(Train[i]),fmt="%s")
		np.savetxt(datasetfolder+'/Test'+str(i)+'.txt', np.asarray(Test[i]),fmt="%s")
		np.savetxt(datasetfolder+'/Val'+str(i)+'.txt', np.asarray(Val[i]),fmt="%s")
	
	return Train, Test, Val

def KFoldSampling(ids, k):
	kf = KFold(n_splits=k, shuffle=True, random_state=1234)
	folds = kf.split(ids)
	train_fold_ids = OrderedDict()
	val_fold_ids = OrderedDict()
	test_fold_ids = OrderedDict()
	for i, (train_indices, test_indices) in enumerate(folds):
		size_all = len(train_indices)
		train_fold_ids[i] = []
		val_fold_ids[i] = []
		test_fold_ids[i] = []
		train_indices2 = train_indices[:int(size_all * 0.8)]
		val_indices = train_indices[int(size_all * 0.8):]

		for s in train_indices2:
			train_fold_ids[i].append(ids[s])

		for s in val_indices:
			val_fold_ids[i].append(ids[s])

		for s in test_indices:
			test_fold_ids[i].append(ids[s])

	return train_fold_ids, val_fold_ids, test_fold_ids

def label_dist(dist):
	return [int(x) for x in dist]

def preprocess_data(left=3999, right=3999, k_fold=8, test_only=False):
	data = Genedata.load_sequence(
		dataset=DATA_FILE,
		left=left,
		right=right,
		predict=False,
	)
	id_label_seq_dict = get_id_label_seq_Dict(data)
	label_id_dict = get_label_id_Dict(id_label_seq_dict)
	Train, Test, Val = group_sample(label_id_dict, DATA_PATH, k_fold)

	X_train, X_test, X_val = {}, {}, {}
	Y_train, Y_test, Y_val = {}, {}, {}

	# Initialize RNA-FM model
	rna_fm = RNA_FM()

	if test_only:
		for i in tqdm(range(len(Test)), desc="Processing Test-Only Folds"):  # Fold num
			tqdm.write(f"Processing data for test-only fold {i+1} (One-Hot Encoding and RNA-FM Embeddings)")
			# Test data
			X_test[i] = []
			for id in tqdm(Test[i], desc=f"Processing Test Data for Fold {i+1}"):
				seq_left = list(id_label_seq_dict[id].values())[0][0]
				seq_right = list(id_label_seq_dict[id].values())[0][1]
				# Pad sequences
				seq_left = seq_left.ljust(left, '-')
				seq_right = seq_right.rjust(right, '-')
				seq_left = ''.join([c if c in encoding_seq.keys() else 'N' for c in seq_left])
				seq_right = ''.join([c if c in encoding_seq.keys() else 'N' for c in seq_right])
				# Combine sequences
				combined_sequence = seq_left + seq_right
				# RNA-FM embedding
				rnafm_embedding = rna_fm.embeddings([(id, combined_sequence)])
				# One-hot encoding
				one_hot_left = [encoding_seq[c] for c in seq_left]
				one_hot_right = [encoding_seq[c] for c in seq_right]
				one_hot_combined = np.array(one_hot_left + one_hot_right).T
				# Append combined sequence, RNA-FM embedding, and one-hot encoding
				X_test[i].append((combined_sequence, rnafm_embedding, one_hot_combined))
			Y_test[i] = torch.tensor([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Test[i]], dtype=torch.float32)
		return X_test, Y_test
	
	for i in tqdm(range(len(Train)), desc="Processing Folds"):  # Fold num
		tqdm.write(f"Processing data for fold {i+1} (One-Hot Encoding and RNA-FM Embeddings)")

		# Train data
		X_train[i] = []
		for id in tqdm(Train[i], desc=f"Processing Train Data for Fold {i+1}"):
			seq_left = list(id_label_seq_dict[id].values())[0][0]
			seq_right = list(id_label_seq_dict[id].values())[0][1]
			# Pad sequences
			seq_left = seq_left.ljust(left, '-')
			seq_right = seq_right.rjust(right, '-')
			seq_left = ''.join([c if c in encoding_seq.keys() else 'N' for c in seq_left])
			seq_right = ''.join([c if c in encoding_seq.keys() else 'N' for c in seq_right])
			# Combine sequences
			combined_sequence = seq_left + seq_right
			# RNA-FM embedding
			rnafm_embedding = rna_fm.embeddings([(id, combined_sequence)])
			# One-hot encoding
			one_hot_left = [encoding_seq[c] for c in seq_left]
			one_hot_right = [encoding_seq[c] for c in seq_right]
			one_hot_combined = np.array(one_hot_left + one_hot_right).T  # Shape: [4, seq_len]
			# Append combined sequence, RNA-FM embedding, and one-hot encoding
			X_train[i].append((combined_sequence, rnafm_embedding, one_hot_combined))
			
		Y_train[i] = torch.tensor([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Train[i]], dtype=torch.float32)

		# Test data
		X_test[i] = []
		for id in tqdm(Test[i], desc=f"Processing Test Data for Fold {i+1}"):
			seq_left = list(id_label_seq_dict[id].values())[0][0]
			seq_right = list(id_label_seq_dict[id].values())[0][1]
			# Pad sequences
			seq_left = seq_left.ljust(left, '-')
			seq_right = seq_right.rjust(right, '-')
			seq_left = ''.join([c if c in encoding_seq.keys() else 'N' for c in seq_left])
			seq_right = ''.join([c if c in encoding_seq.keys() else 'N' for c in seq_right])
			# Combine sequences
			combined_sequence = seq_left + seq_right
			# RNA-FM embedding
			rnafm_embedding = rna_fm.embeddings([(id, combined_sequence)])
			# One-hot encoding
			one_hot_left = [encoding_seq[c] for c in seq_left]
			one_hot_right = [encoding_seq[c] for c in seq_right]
			one_hot_combined = np.array(one_hot_left + one_hot_right).T  # Shape: [4, seq_len]
			# Append combined sequence, RNA-FM embedding, and one-hot encoding
			X_test[i].append((combined_sequence, rnafm_embedding, one_hot_combined))

		Y_test[i] = torch.tensor([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Test[i]], dtype=torch.float32)

		# Validation data
		X_val[i] = []
		for id in tqdm(Val[i], desc=f"Processing Validation Data for Fold {i+1}"):
			seq_left = list(id_label_seq_dict[id].values())[0][0]
			seq_right = list(id_label_seq_dict[id].values())[0][1]
			# Pad sequences
			seq_left = seq_left.ljust(left, '-')
			seq_right = seq_right.rjust(right, '-')
			seq_left = ''.join([c if c in encoding_seq.keys() else 'N' for c in seq_left])
			seq_right = ''.join([c if c in encoding_seq.keys() else 'N' for c in seq_right])
			# Combine sequences
			combined_sequence = seq_left + seq_right
			# RNA-FM embedding
			rnafm_embedding = rna_fm.embeddings([(id, combined_sequence)])
			# One-hot encoding
			one_hot_left = [encoding_seq[c] for c in seq_left]
			one_hot_right = [encoding_seq[c] for c in seq_right]
			one_hot_combined = np.array(one_hot_left + one_hot_right).T  # Shape: [4, seq_len]
			# Append combined sequence, RNA-FM embedding, and one-hot encoding
			X_val[i].append((combined_sequence, rnafm_embedding, one_hot_combined))

		Y_val[i] = torch.tensor([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Val[i]], dtype=torch.float32)

	return X_train, X_test, X_val, Y_train, Y_test, Y_val

def train_model(model, mname, X_train, Y_train, X_test, Y_test, X_val,
				Y_val, batch_size, epochs, lr=0.001, weight_decay=1e-4, save_path="./models", log_file="training_log.txt", load_CNN_model_path=None):
	""" Train the LLM/CNN model and write logs to a file """
	model = model.to(DEVICE)

	# Load the CNN model here
	cnn_model = MultiscaleCNNModel(
		layers=MultiscaleCNNLayers(
			in_channels=64,
			embedding_dim=4,  # For one-hot encoding
			pooling_size=8,
			pooling_stride=8,
			drop_rate_cnn=0.3,
			drop_rate_fc=0.3,
			length=7998,  # Length of the input sequence
			nb_classes=7
		),
		num_classes=7,
		sequence_length=15,
		aggregation_method="learnable_linear"
	).to(DEVICE)
	cnn_model.load_state_dict(torch.load(load_CNN_model_path))
	cnn_model.eval()

	# Load the RNA-FM model
	rna_fm = RNA_FM()

	# Replace BCEWithLogitsLoss with StableFocalLoss
	alpha = torch.tensor([0.42, 0.01, 0.76, 0.64, 0.63, 0.66, 1.0]).to(DEVICE)  # * number of classes (using the EDC paper first, )
	gamma = 0 # paper value
	omega = torch.tensor([1.3, 1.0, 0.9, 1.4, 1.2, 1.4, 1.0]).to(DEVICE)  # * number of classes
	criterion = StableFocalLoss(alpha=alpha, gamma=gamma, omega=omega, reduction='mean')
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	
	os.makedirs(save_path, exist_ok=True)
	
	# Initialize metrics for aggregation
	thresholds = np.linspace(0, 1, 101)
	average_metrics = {"MCC": np.zeros((Y_test[0].shape[1], len(thresholds))),
					   "Precision": np.zeros((Y_test[0].shape[1], len(thresholds))),
					   "Recall": np.zeros((Y_test[0].shape[1], len(thresholds)))}

	
	# Open the log file
	with open(log_file, "w") as log:
		for i in tqdm(range(len(X_train)), desc="Training Folds"):  # Fold num
			log.write(f"fold {i+1} ({mname} Training)\n")
			tqdm.write(f"fold {i+1} ({mname} Training)")
			early_stopper = EarlyStopper(patience=3, min_delta=0.05)
			# Create DataLoaders
			train_dataset = GeneDataset(X_train[i], Y_train[i])
			val_dataset = GeneDataset(X_val[i], Y_val[i])
			test_dataset = GeneDataset(X_test[i], Y_test[i])
			train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
			val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
			test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

			best_val_loss = float('inf')

			for epoch in range(epochs):
				# Training step
				model.train()
				train_loss = 0
				for combined_sequence, rnafm_embedding, one_hot_encoding, labels in train_loader:
					rnafm_embeddings, one_hot_encoding, labels = (
						rnafm_embedding.to(DEVICE),
						one_hot_encoding.to(DEVICE).float(),
						labels.to(DEVICE).float(),
					)
					optimizer.zero_grad()
					with torch.no_grad():
						cnn_embeddings, _ = cnn_model(one_hot_encoding)
					# Concatenate CNN embeddings with RNA-FM model output
					outputs = model(cnn_embeddings, rnafm_embeddings)
					loss = criterion(outputs, labels)
					loss.backward()
					optimizer.step()
					train_loss += loss.item()

				train_loss /= len(train_loader)

				# Validation step
				model.eval()
				with torch.no_grad():
					val_loss = 0
					for combined_sequence, rnafm_embedding, one_hot_encoding, labels in val_loader:
						rnafm_embeddings, one_hot_encoding, labels = (
							rnafm_embedding.to(DEVICE),
							one_hot_encoding.to(DEVICE).float(),
							labels.to(DEVICE).float(),
						)
						with torch.no_grad():
							cnn_embeddings, _ = cnn_model(one_hot_encoding)
						# Concatenate CNN embeddings with RNA-FM model output
						outputs = model(cnn_embeddings, rnafm_embeddings)
						loss = criterion(outputs, labels)
						val_loss += loss.item()

				val_loss /= len(val_loader)

				if early_stopper.early_stop(val_loss):  # Early stopper triggers
					log_message = f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}\nEarly Stop Trigger.\n"
					log.write(log_message)
					tqdm.write(log_message)
					break
				
				log_message = f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}\n"
				log.write(log_message)
				tqdm.write(log_message)

				# Save the best model
				if val_loss < best_val_loss:
					best_val_loss = val_loss
					model_save_path = os.path.join(save_path, f"{mname}_model_fold{i+1}.pth")
					torch.save(model.state_dict(), model_save_path)
					log.write(f"Best model for fold {i+1} saved with Val Loss: {best_val_loss:.5f}\n")
					tqdm.write(f"Best model for fold {i+1} saved with Val Loss: {best_val_loss:.5f}")

			# Load the best model for this fold
			best_model_path = os.path.join(save_path, f"{mname}_model_fold{i+1}.pth")
			model.load_state_dict(torch.load(best_model_path), strict=False)
			log.write(f"Best model for fold {i+1} loaded from {best_model_path}\n")
			tqdm.write(f"Best model for fold {i+1} loaded from {best_model_path}")

			# Test step
			model.eval()
			test_loss = 0
			with torch.no_grad():
				all_labels = []
				all_outputs = []
				for combined_sequence, rnafm_embedding, one_hot_encoding, labels in test_loader:
					rnafm_embeddings, one_hot_encoding, labels = (
						rnafm_embedding.to(DEVICE),
						one_hot_encoding.to(DEVICE).float(),
						labels.to(DEVICE).float(),
					)
					optimizer.zero_grad()
					with torch.no_grad():
						cnn_embeddings, _ = cnn_model(one_hot_encoding)
					# Concatenate CNN embeddings with RNA-FM model output
					outputs = model(cnn_embeddings, rnafm_embeddings)
					loss = criterion(outputs, labels)
					all_labels.append(labels.cpu().numpy())
					all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
					test_loss += loss.item()

				test_loss /= len(test_loader)
				log.write(f"Test Loss: {test_loss:.5f}\n")  # Log test loss
				tqdm.write(f"Test Loss: {test_loss:.5f}")  # Print test loss

				all_labels = np.concatenate(all_labels, axis=0)
				all_outputs = np.concatenate(all_outputs, axis=0)

				# Aggregate metrics across folds
				for class_idx in range(Y_test[i].shape[1]):
					true_labels = all_labels[:, class_idx]
					pred_probs = all_outputs[:, class_idx]

					for t_idx, threshold in enumerate(thresholds):
						pred_labels = (pred_probs >= threshold).astype(int)
						mcc = matthews_corrcoef(true_labels, pred_labels) if len(np.unique(true_labels)) > 1 else np.nan
						precision = average_precision_score(true_labels, pred_labels) if len(np.unique(true_labels)) > 1 else np.nan
						recall = np.sum(np.logical_and(pred_labels, true_labels)) / np.sum(true_labels) if np.sum(true_labels) > 0 else 0

						if not np.isnan(mcc):
							average_metrics["MCC"][class_idx, t_idx] += mcc
							average_metrics["Precision"][class_idx, t_idx] += precision
							average_metrics["Recall"][class_idx, t_idx] += recall

		# Average metrics over all folds
		for key in average_metrics:
			average_metrics[key] /= len(X_train)

		# Find the best MCC and its threshold for each class
		best_thresholds = []
		for class_idx in range(Y_test[0].shape[1]):
			best_mcc_idx = np.argmax(average_metrics["MCC"][class_idx])
			best_threshold = thresholds[best_mcc_idx]
			best_thresholds.append(best_threshold)
			log.write(f"Class {class_idx + 1} - Best MCC: {average_metrics['MCC'][class_idx, best_mcc_idx]:.5f} at Threshold: {best_threshold:.2f}\n")
			tqdm.write(f"Class {class_idx + 1} - Best MCC: {average_metrics['MCC'][class_idx, best_mcc_idx]:.5f} at Threshold: {best_threshold:.2f}")

		# Plot average metrics vs thresholds for each class
		for class_idx in range(Y_test[0].shape[1]):
			plt.figure()
			plt.plot(thresholds, average_metrics["MCC"][class_idx], label="MCC")
			plt.plot(thresholds, average_metrics["Precision"][class_idx], label="Precision")
			plt.plot(thresholds, average_metrics["Recall"][class_idx], label="Recall")
			plt.axvline(x=best_thresholds[class_idx], color='r', linestyle='--', label=f"Best Threshold: {best_thresholds[class_idx]:.2f}")
			plt.xlabel("Threshold")
			plt.ylabel("Metrics")
			plt.title(f"Class {class_idx + 1} Average Metrics vs Threshold")
			plt.legend()
			plt.grid()
			plt.savefig(os.path.join(f"./plots", f"{mname}_class_{class_idx + 1}_average_metrics_vs_threshold.png"))
			plt.close()

	return

def test_model(path1, path2, X_test, Y_test, k_fold, batch_size=32, thresholds=None, log_file="test_log.txt"):

	# Load the CNN model n model weights
	cnn_model = MultiscaleCNNModel(
		layers=MultiscaleCNNLayers(
			in_channels=64,
			embedding_dim=4,  # For one-hot encoding
			pooling_size=8,
			pooling_stride=8,
			drop_rate_cnn=0.3,
			drop_rate_fc=0.3,
			length=7998,  # Length of the input sequence
			nb_classes=7
		),
		num_classes=7,
		sequence_length=15,
		aggregation_method="learnable_linear"
	).to(DEVICE)
	cnn_model.load_state_dict(torch.load(path2))

	# Load the combined model n model weights
	model = CombinedModel(
		cnn_output_dim=7,
		llm_output_dim=640,
		hidden_dim=32,
		nb_classes=7
	).to(DEVICE)
	model.load_state_dict(torch.load(path1), strict=False)

	model.eval()
	cnn_model.eval()

	# Default thresholds if not provided
	if thresholds is None:
		thresholds = [0.5] * Y_test[0].shape[1]  # Default threshold of 0.5 for each class

	# Initialize metrics
	overall_metrics = {"AUC-ROC": np.zeros(Y_test[0].shape[1]),
					   "AUC-PR": np.zeros(Y_test[0].shape[1]),
					   "MCC": np.zeros(Y_test[0].shape[1])}

	with open(log_file, "w") as log:
		log.write(f"Testing the combined model with {k_fold}-folds:\n")
		tqdm.write("Testing the combined model:")

		for fold_idx in range(len(X_test)):
			test_dataset = GeneDataset(X_test[fold_idx], Y_test[fold_idx])
			test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

			all_labels = []
			all_outputs = []

			with torch.no_grad():
				for combined_sequence, rnafm_embedding, one_hot_encoding, labels in test_loader:
					# Move data to device
					rnafm_embedding, one_hot_encoding, labels = (
						rnafm_embedding.to(DEVICE),
						one_hot_encoding.to(DEVICE).float(),
						labels.to(DEVICE).float(),
					)

					cnn_embeddings, _ = cnn_model(one_hot_encoding)
					outputs = model(cnn_embeddings, rnafm_embedding)
					# Store labels and predictions
					all_labels.append(labels.cpu().numpy())
					all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

			# Concatenate all labels and outputs
			all_labels = np.concatenate(all_labels, axis=0)
			all_outputs = np.concatenate(all_outputs, axis=0)

			# Compute metrics for each class
			for class_idx in range(all_labels.shape[1]):
				true_labels = all_labels[:, class_idx]
				pred_probs = all_outputs[:, class_idx]

				# AUC-ROC
				aucroc = roc_auc_score(true_labels, pred_probs) if len(np.unique(true_labels)) > 1 else np.nan
				overall_metrics["AUC-ROC"][class_idx] += aucroc if not np.isnan(aucroc) else 0
				# AUC-PR
				aucpr = average_precision_score(true_labels, pred_probs) if len(np.unique(true_labels)) > 1 else np.nan
				overall_metrics["AUC-PR"][class_idx] += aucpr if not np.isnan(aucpr) else 0
				# MCC
				pred_labels = (pred_probs >= thresholds[class_idx]).astype(int)
				mcc = matthews_corrcoef(true_labels, pred_labels) if len(np.unique(true_labels)) > 1 else np.nan
				overall_metrics["MCC"][class_idx] += mcc if not np.isnan(mcc) else 0

		# Average metrics across folds
		for key in overall_metrics:
			overall_metrics[key] /= len(X_test)

		# Log metrics
		log.write("Overall Metrics:\n")
		tqdm.write("Overall Metrics:")
		for class_idx in range(Y_test[0].shape[1]):
			log.write(f"Class {class_idx + 1} - AUC-ROC: {overall_metrics['AUC-ROC'][class_idx]:.5f}, "
					  f"AUC-PR: {overall_metrics['AUC-PR'][class_idx]:.5f}, "
					  f"MCC: {overall_metrics['MCC'][class_idx]:.5f}\n")
			tqdm.write(f"Class {class_idx + 1} - AUC-ROC: {overall_metrics['AUC-ROC'][class_idx]:.5f}, "
					   f"AUC-PR: {overall_metrics['AUC-PR'][class_idx]:.5f}, "
					   f"MCC: {overall_metrics['MCC'][class_idx]:.5f}")

	return

def optimize_thresholds_and_plot(model, test_loader, class_count, save_path="./plots"):
	"""
	Optimize thresholds for each class and plot MCC, Precision, and Recall vs thresholds.
	"""
	if save_path is None:
		save_path = "./plots"  # Default to "./plots" if save_path is None
	os.makedirs(save_path, exist_ok=True)

	thresholds = np.linspace(0, 1, 101)
	metrics = {"MCC": [], "Precision": [], "Recall": []}

	for class_idx in range(class_count):
		mcc_values, precision_values, recall_values = [], [], []

		for threshold in thresholds:
			all_labels, all_preds = [], []

			with torch.no_grad():
				for sequences, labels in test_loader:
					sequences, labels = sequences.to(DEVICE), labels.to(DEVICE).float()
					outputs = torch.sigmoid(model(sequences))
					preds = (outputs[:, class_idx] >= threshold).cpu().numpy()
					all_labels.extend(labels[:, class_idx].cpu().numpy())
					all_preds.extend(preds)

			# Calculate metrics
			mcc = matthews_corrcoef(all_labels, all_preds)
			precision = average_precision_score(all_labels, all_preds)
			recall = np.sum(np.logical_and(all_preds, all_labels)) / np.sum(all_labels)

			mcc_values.append(mcc)
			precision_values.append(precision)
			recall_values.append(recall)

		# Store metrics for plotting
		metrics["MCC"].append(mcc_values)
		metrics["Precision"].append(precision_values)
		metrics["Recall"].append(recall_values)

		# Plot metrics vs thresholds
		plt.figure()
		plt.plot(thresholds, mcc_values, label="MCC")
		plt.plot(thresholds, precision_values, label="Precision")
		plt.plot(thresholds, recall_values, label="Recall")
		plt.xlabel("Threshold")
		plt.ylabel("Metrics")
		plt.title(f"Class {class_idx + 1} Metrics vs Threshold")
		plt.legend()
		plt.savefig(os.path.join(save_path, f"class_{class_idx + 1}_metrics.png"))
		plt.close()

	return metrics

if __name__ == "__main__":

	# Test the model
	if os.path.exists("./llm_cnn_test_data.pth"):
		data = torch.load("./llm_cnn_test_data.pth")
		X_test = data["X_test"]
		Y_test = data["Y_test"]
	else:
		X_test, Y_test = preprocess_data(
			left=4000,
			right=4000,
			k_fold=5,
			test_only=True
		)	
		torch.save({
			"X_test": X_test,
			"Y_test": Y_test,
		}, "llm_cnn_test_data.pth")

	print("Shape of X_test:", len(X_test))
	print("Shape of Y_test:", len(Y_test))
	
	test_model(
		path1="./LLM_CNN_model/LLM_CNN_model_model_fold8.pth",
		path2="./cnn_models_linear_custom_loss/CNN_Linear_CustomLoss_model_fold8.pth",
		X_test=X_test,
		Y_test=Y_test,
		k_fold=5,
		batch_size=32,
		#thresholds=None,
		thresholds=[0.65, 0.94, 0.21, 0.56, 0.34, 0.29, 0.14],
		log_file="test_result_log.txt"
	)