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

from gene_data import Genedata
from hier_attention_mask import AttentionMask
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
DATA_PATH = "./data/"
DATA_FILE = DATA_PATH + "modified_multilabel_seq_nonredundent.fasta"
#DATA_FILE = DATA_PATH + "test.fasta"

CNN_EPOCHS = 20
# Update the THRESHOLD global variable with class-specific thresholds

#THRESHOLD = [0.60, 0.97, 0.22, 0.48, 0.26, 0.23, 0.15] #CNN LINEAR
THRESHOLD = [0.59, 0.73, 0.16, 0.51, 0.36, 0.25, 0.18] #CNN MEAN

encoding_seq = OrderedDict([
	('A', [1, 0, 0, 0]),
	('G', [0, 0, 1, 0]),
	('T', [0, 0, 0, 1]),
	('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
	('-', [0, 0, 0, 0]),  # Pad
])

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

# Load mRNA-FM model (LLM)
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

		for label, sequence in tqdm(data, desc="Generating embeddings", unit="sequence"):
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
			all_embeddings.append(combined_embedding)
			print(all_embeddings[-1].shape) # [1, 8000, 640]
			exit()

		# Concatenate all embeddings along the batch dimension
		return torch.cat(all_embeddings, dim=0)  # Shape: [batch_size, seq_len, 1280]
			
class LLMClassifier(nn.Module):
	def __init__(self, output_dim):
		super(LLMClassifier, self).__init__()
		self.output_dim = output_dim

		self.fc1 = nn.Linear(1280, 512)
		self.fc2 = nn.Linear(512, 64)
		self.fc3 = nn.Linear(64, output_dim)
		self.activation = nn.GELU()
		self.dropout = nn.Dropout(0.1)

	def forward(self, x):
		x = self.fc1(x)
		x = self.activation(x)
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.activation(x)
		x = self.dropout(x)
		x = self.fc3(x)
		return x

class MultiscaleCNNLayers(nn.Module):
	def __init__(self, in_channels, embedding_dim, pooling_size, pooling_stride, drop_rate_cnn, drop_rate_fc, length, nb_classes):
		super(MultiscaleCNNLayers, self).__init__()

		self.bn1 = nn.BatchNorm1d(in_channels)
		self.bn2 = nn.BatchNorm1d(in_channels // 2)

		# Explicit padding calculation: (kernel_size - 1) // 2
		self.conv1_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=in_channels, kernel_size=9, padding=4)
		self.conv1_2 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=9, padding=4)
		self.conv2_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=in_channels, kernel_size=20, padding=10)
		self.conv2_2 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=20, padding=10)
		self.conv3_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=in_channels // 2, kernel_size=49, padding=24)
		self.conv3_2 = nn.Conv1d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=49, padding=24)

		self.pool = nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_stride)

		self.dropout_cnn = nn.Dropout(drop_rate_cnn)
		self.dropout_fc = nn.Dropout(drop_rate_fc)

		self.fc = nn.Linear(length // pooling_stride, nb_classes)
		nn.init.xavier_uniform_(self.fc.weight)

		self.activation = nn.GELU() 

		self.attentionHead = AttentionMask(hidden = (length // pooling_stride), da = 80, r = 5)

	def forward_cnn(self, x, conv1, conv2, bn1, bn2):
		#print(x.shape) #torch.Size([32, 4, 8000])
		x = conv1(x)
		#print(x.shape) #torch.Size([32, 64, 8000])
		x = self.activation(bn1(x))
		#print(x.shape) #torch.Size([32, 64, 8000])
		x = conv2(x)
		#print(x.shape) #torch.Size([32, 32, 8000])
		x = self.activation(bn2(x))
		#print(x.shape) #torch.Size([32, 32, 8000])
		x = self.pool(x)
		#print(x.shape) #torch.Size([32, 32, 1000])
		x, regulatization_loss, att_map = self.attentionHead(x)
		#print(x.shape) #torch.Size([32, 5, 999])
		x = self.dropout_cnn(x)
		return x, regulatization_loss
		#return x, regulatization_loss, att_map
	
class MultiscaleCNNModel(nn.Module):
	def __init__(self, layers, num_classes, sequence_length, aggregation_method="mean"):
		super(MultiscaleCNNModel, self).__init__()
		self.layers = layers
		self.aggregation_method = aggregation_method  # "mean" or "learnable_weights"
		
		self.fc1 = nn.Linear(1280, 512)
		self.fc2 = nn.Linear(512, 64)
		self.fc3 = nn.Linear(64, 4)
		self.dropout = nn.Dropout(0.1)

		if aggregation_method == "learnable_linear":
			self.sequence_aggregator = nn.Linear(sequence_length, 1)

	def forward(self, x):
		#x = self.dropout(self.fc1(x))
		#x = self.dropout(self.fc2(x))
		#x = self.fc3(x)
		#print(x.shape)
		#exit()
		#x = x.permute(0, 2, 1)  # Shape: [batch_size, num_classes, sequence_length]
		#print(x.shape)
		# Pass through CNN layers
		x1, x1_regulatization_loss = self.layers.forward_cnn(x, self.layers.conv1_1, self.layers.conv1_2, self.layers.bn1, self.layers.bn2)
		x2, x2_regulatization_loss = self.layers.forward_cnn(x, self.layers.conv2_1, self.layers.conv2_2, self.layers.bn1, self.layers.bn2)
		x3, x3_regulatization_loss = self.layers.forward_cnn(x, self.layers.conv3_1, self.layers.conv3_2, self.layers.bn2, self.layers.bn2)
		x = torch.cat((x1, x2, x3), dim=1) # [32, 15, 999]
		x = self.layers.dropout_fc(self.layers.fc(x)) # [32, 15, 7]
		if self.aggregation_method == "mean":
			# [32, 7]
			x = torch.mean(x, dim=1)  # Shape: [batch_size, concat_channels, num_classes] -> [batch_size, num_classes]
		elif self.aggregation_method == "learnable_linear":
			# [32, 7]
			x = x.permute(0, 2, 1)  # Shape: [batch_size, 7, 15]
			x = self.sequence_aggregator(x).squeeze(-1) 

		return x, x1_regulatization_loss + x2_regulatization_loss + x3_regulatization_loss
	
#class EnsembleModel(nn.Module):
#	def __init__(self, llm_model, cnn_model, llm_output_dim, cnn_output_dim, hidden_dim, nb_classes):
#		super(EnsembleModel, self).__init__()
#		self.llm_model = llm_model
#		self.cnn_model = cnn_model

#		# Fully connected NN for combining LLM and CNN outputs and length compoent in second layer
#		self.fc1 = nn.Linear(llm_output_dim + cnn_output_dim, (llm_output_dim + cnn_output_dim)/ 2 + 1)
#		self.fc2 = nn.Linear((llm_output_dim + cnn_output_dim)/ 2 + 1, nb_classes)
#		self.activation = nn.ReLU()
#		self.dropout = nn.Dropout(0.2)
	
#	def forward(self, x):
#		llm_output = self.llm_model(x) # LLM output
#		cnn_output = self.cnn_model(x) # CNN output

#		x = torch.cat((llm_output, cnn_output), dim=1) # Combine LLM and CNN outputs

#		x = self.fc1(x)
#		self.activation(x)
#		x = self.dropout(x)
#		x = self.fc2(x)

#		return x

class GeneDataset(Dataset):
	def __init__(self, sequences, labels):
		self.sequences = sequences
		self.labels = labels

	def __len__(self):
		return len(self.sequences)

	def __getitem__(self, idx):
		return self.sequences[idx], self.labels[idx]
	
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
			#print('label:%s finished sampling! Train length: %s, Test length: %s, Val length:%s'%(eachkey, len(train_fold_ids[i]), len(test_fold_ids[i]),len(val_fold_ids[i])))
	
	for i in range(foldnum):
		print('Train length: %s, Test length: %s, Val length: %s'%(len(Train[i]),len(Test[i]),len(Val[i])))
		#print(type(Train[i]))
		#print(Train[0][:foldnum])
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

def preprocess_data_onehot(left=3999, right=3999, k_fold=8):
	# Prepare data
	data = Genedata.load_sequence(
		dataset=DATA_FILE,
		left=left, # divsible by 3
		right=right,
		predict=False,
	)
	id_label_seq_dict = get_id_label_seq_Dict(data)
	label_id_dict = get_label_id_Dict(id_label_seq_dict)
	Train, Test, Val = group_sample(label_id_dict, DATA_PATH, k_fold)

	X_train, X_test, X_val = {}, {}, {}
	Y_train, Y_test, Y_val = {}, {}, {}
	for i in tqdm(range(len(Train))):  # Fold num
		tqdm.write(f"Padding and Indexing data for fold {i+1} (One-Hot Encoding)")
		seq_encoding_keys = list(encoding_seq.keys())
		seq_encoding_vectors = np.array(list(encoding_seq.values()))

		# Train data
		X_train[i] = []
		for id in Train[i]:
			seq_left = list(id_label_seq_dict[id].values())[0][0]
			seq_right = list(id_label_seq_dict[id].values())[0][1]
			# Pad sequences
			seq_left = seq_left.ljust(left, '-')
			seq_right = seq_right.rjust(right, '-')
			seq_left = ''.join([c if c in seq_encoding_keys else 'N' for c in seq_left])
			seq_right = ''.join([c if c in seq_encoding_keys else 'N' for c in seq_right])
			# One-hot encode
			one_hot_left = [seq_encoding_vectors[seq_encoding_keys.index(c)] for c in seq_left]
			one_hot_right = [seq_encoding_vectors[seq_encoding_keys.index(c)] for c in seq_right]
			# Combine left and right
			combined = np.concatenate([one_hot_left, one_hot_right], axis=0)
			X_train[i].append(combined)

		# Convert list of NumPy arrays to a single NumPy array, then to a PyTorch tensor
		X_train[i] = torch.tensor(np.array(X_train[i]), dtype=torch.float32).permute(0, 2, 1)
		Y_train[i] = torch.tensor([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Train[i]], dtype=torch.float32)

		# Test data
		X_test[i] = []
		for id in Test[i]:
			seq_left = list(id_label_seq_dict[id].values())[0][0]
			seq_right = list(id_label_seq_dict[id].values())[0][1]
			# Pad sequences
			seq_left = seq_left.ljust(left, '-')
			seq_right = seq_right.rjust(right, '-')
			seq_left = ''.join([c if c in seq_encoding_keys else 'N' for c in seq_left])
			seq_right = ''.join([c if c in seq_encoding_keys else 'N' for c in seq_right])
			# One-hot encode
			one_hot_left = [seq_encoding_vectors[seq_encoding_keys.index(c)] for c in seq_left]
			one_hot_right = [seq_encoding_vectors[seq_encoding_keys.index(c)] for c in seq_right]
			# Combine left and right
			combined = np.concatenate([one_hot_left, one_hot_right], axis=0)
			X_test[i].append(combined)

		# Convert list of NumPy arrays to a single NumPy array, then to a PyTorch tensor
		X_test[i] = torch.tensor(np.array(X_test[i]), dtype=torch.float32).permute(0, 2, 1)
		Y_test[i] = torch.tensor([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Test[i]], dtype=torch.float32)

		# Validation data
		X_val[i] = []
		for id in Val[i]:
			seq_left = list(id_label_seq_dict[id].values())[0][0]
			seq_right = list(id_label_seq_dict[id].values())[0][1]
			# Pad sequences
			seq_left = seq_left.ljust(left, '-')
			seq_right = seq_right.rjust(right, '-')
			seq_left = ''.join([c if c in seq_encoding_keys else 'N' for c in seq_left])
			seq_right = ''.join([c if c in seq_encoding_keys else 'N' for c in seq_right])
			# One-hot encode
			one_hot_left = [seq_encoding_vectors[seq_encoding_keys.index(c)] for c in seq_left]
			one_hot_right = [seq_encoding_vectors[seq_encoding_keys.index(c)] for c in seq_right]
			# Combine left and right
			combined = np.concatenate([one_hot_left, one_hot_right], axis=0)
			X_val[i].append(combined)

		# Convert list of NumPy arrays to a single NumPy array, then to a PyTorch tensor
		X_val[i] = torch.tensor(np.array(X_val[i]), dtype=torch.float32).permute(0, 2, 1)
		Y_val[i] = torch.tensor([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Val[i]], dtype=torch.float32)
	
	return X_train, X_test, X_val, Y_train, Y_test, Y_val

def preprocess_data_raw_with_embeddings(left=3999, right=3999):
	"""
	Prepare raw data for LLM training and generate embeddings using mRNA_FM.
	"""
	# Load raw data
	data = Genedata.load_sequence(
		dataset=DATA_FILE,
		left=left,  # Divisible by 3
		right=right,
		predict=False,
	)
	id_label_seq_dict = get_id_label_seq_Dict(data)
	label_id_dict = get_label_id_Dict(id_label_seq_dict)
	Train, Test, Val = group_sample(label_id_dict, DATA_PATH)

	X_train, X_test, X_val = {}, {}, {}
	Y_train, Y_test, Y_val = {}, {}, {}

	# Initialize mRNA_FM model
	mrna_fm = RNA_FM()

	for i in tqdm(range(len(Train))):  # Fold num
		tqdm.write(f"Preparing raw data and generating embeddings for fold {i+1}")

		# Train data
		train_data = [
			(
				"id_" + str(idx),
				(list(id_label_seq_dict[id].values())[0][0] + list(id_label_seq_dict[id].values())[0][1])
			)
			for idx, id in enumerate(Train[i])
		]
		X_train[i] = torch.tensor(mrna_fm.embeddings(train_data).cpu().numpy(), dtype=torch.float32)  # Convert to Tensor
		Y_train[i] = torch.tensor([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Train[i]], dtype=torch.float32)

		# Test data
		test_data = [
			(
				"id_" + str(idx),
				(list(id_label_seq_dict[id].values())[0][0] + list(id_label_seq_dict[id].values())[0][1])
			)
			for idx, id in enumerate(Test[i])
		]
		X_test[i] = torch.tensor(mrna_fm.embeddings(test_data).cpu().numpy(), dtype=torch.float32)  # Convert to Tensor
		Y_test[i] = torch.tensor([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Test[i]], dtype=torch.float32)

		# Validation data
		val_data = [
			(
				"id_" + str(idx),
				(list(id_label_seq_dict[id].values())[0][0] + list(id_label_seq_dict[id].values())[0][1])
			)
			for idx, id in enumerate(Val[i])
		]
		X_val[i] = torch.tensor(mrna_fm.embeddings(val_data).cpu().numpy(), dtype=torch.float32)  # Convert to Tensor
		Y_val[i] = torch.tensor([label_dist(list(id_label_seq_dict[id].keys())[0]) for id in Val[i]], dtype=torch.float32)

	return X_train, X_test, X_val, Y_train, Y_test, Y_val

def train_model(model, mname, X_train, Y_train, X_test, Y_test, X_val,
				Y_val, batch_size, epochs, lr=0.001, weight_decay=5e-5, save_path="./models", log_file="training_log.txt"):
	""" Train the LLM/CNN model and write logs to a file """
	model = model.to(DEVICE)
	criterion = nn.BCEWithLogitsLoss()  # MultiLabel Classification -> Multi Binary Classification
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	
	os.makedirs(save_path, exist_ok=True)
	
	# Initialize metrics for aggregation
	thresholds = np.linspace(0, 1, 101)
	average_metrics = {"MCC": np.zeros((Y_test[0].shape[1], len(thresholds))),
					   "Precision": np.zeros((Y_test[0].shape[1], len(thresholds))),
					   "Recall": np.zeros((Y_test[0].shape[1], len(thresholds)))}

	# Open the log file
	with open(log_file, "w") as log:
		for i in tqdm(range(len(X_train))):  # Fold num
			log.write(f"fold {i+1} ({mname} Training)\n")
			tqdm.write(f"fold {i+1} ({mname} Training)")
			early_stopper = EarlyStopper(patience=3, min_delta=0.01) 
			# Create DataLoaders for preloaded embeddings
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
				for sequences, labels in train_loader:
					sequences, labels = sequences.to(DEVICE), labels.to(DEVICE).float()
					optimizer.zero_grad()
					outputs, regularization_loss = model(sequences)		
					loss = criterion(outputs, labels) + regularization_loss  # Add regularization loss
					loss.backward()
					optimizer.step()
					train_loss += loss.item()

				train_loss /= len(train_loader)

				# Validation step
				model.eval()
				with torch.no_grad():
					val_loss = 0
					for sequences, labels in val_loader:
						sequences, labels = sequences.to(DEVICE), labels.to(DEVICE).float()
						outputs, regularization_loss = model(sequences)
						loss = criterion(outputs, labels) + regularization_loss
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
				for sequences, labels in test_loader:
					sequences, labels = sequences.to(DEVICE), labels.to(DEVICE).float()
					outputs, regularization_loss = model(sequences)
					all_labels.append(labels.cpu().numpy())
					all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
					loss = criterion(outputs, labels) + regularization_loss  # Compute test loss
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

def test_model(model_path, result_file="result.txt", k_fold=5):
	""" Test the LLM/CNN model with k folds and write result to a file """	
	
	X_train, X_test, X_val, Y_train, Y_test, Y_val = preprocess_data_onehot(
		left=4000,
		right=4000,
		k_fold=5
	)
	
	model_files = [f for f in os.listdir(model_path) if f.endswith(".pth")]
	
	# Initialize arrays to store metrics across all folds
	overall_metrics = {"AUC-ROC": np.zeros(Y_test[0].shape[1]),
					   "AUC-PR": np.zeros(Y_test[0].shape[1]),
					   "MCC": np.zeros(Y_test[0].shape[1])}

	# Open the result file
	with open(result_file, "w") as log:
		log.write(f"Evaluating models in path: {model_path}\n")
		log.write(f"Using {k_fold}-fold cross-validation\n\n")

		for model_file in model_files:
			model_file_path = os.path.join(model_path, model_file)
			log.write(f"Evaluating model over k-folds: {model_file}\n")
			tqdm.write(f"Evaluating model: {model_file}")
			
			layers = MultiscaleCNNLayers(
				in_channels=64,
				embedding_dim=4,  # For one-hot encoding
				pooling_size=8,
				pooling_stride=8,
				drop_rate_cnn=0.3,
				drop_rate_fc=0.3,
				length=7998, # Length of the input sequence
				nb_classes=7
			)

			model = MultiscaleCNNModel(
				layers=layers,
				num_classes=7,
				sequence_length=15, 
				aggregation_method="learnable_linear"
			).to(DEVICE)

			model.load_state_dict(torch.load(model_file_path), strict=False)
			model.eval()

			# Initialize metrics for this model
			average_metrics = {"AUC-ROC": np.zeros(Y_test[0].shape[1]),
							   "AUC-PR": np.zeros(Y_test[0].shape[1]),
							   "MCC": np.zeros(Y_test[0].shape[1])}

			for fold in range(k_fold):
				test_dataset = GeneDataset(X_test[fold], Y_test[fold])
				test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
				all_labels = []
				all_outputs = []

				with torch.no_grad():
					for sequences, labels in test_loader:
						sequences, labels = sequences.to(DEVICE), labels.to(DEVICE).float()
						outputs, _ = model(sequences)
						all_labels.append(labels.cpu().numpy())
						all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

				all_labels = np.concatenate(all_labels, axis=0)
				all_outputs = np.concatenate(all_outputs, axis=0) 

				for class_idx in range(all_labels.shape[1]):
					true_labels = all_labels[:, class_idx]
					pred_probs = all_outputs[:, class_idx]			
					# AUC-ROC
					aucroc = roc_auc_score(true_labels, pred_probs) if len(np.unique(true_labels)) > 1 else np.nan
					average_metrics["AUC-ROC"][class_idx] += aucroc if not np.isnan(aucroc) else 0
					# AUC-PR
					aucpr = average_precision_score(true_labels, pred_probs) if len(np.unique(true_labels)) > 1 else np.nan
					average_metrics["AUC-PR"][class_idx] += aucpr if not np.isnan(aucpr) else 0
					# MCC
					pred_labels = (pred_probs >= THRESHOLD[class_idx]).astype(int)
					mcc = matthews_corrcoef(true_labels, pred_labels) if len(np.unique(true_labels)) > 1 else np.nan
					average_metrics["MCC"][class_idx] += mcc if not np.isnan(mcc) else 0

			# Divide by the number of folds to compute the average for this model
			for key in average_metrics:
				average_metrics[key] /= k_fold
			
			for key in overall_metrics:
				overall_metrics[key] += average_metrics[key]

			log.write(f"Metrics for model: {model_file}\n")
			for class_idx in range(Y_test[0].shape[1]):
				log.write(f"Class {class_idx + 1} - AUC-ROC: {average_metrics['AUC-ROC'][class_idx]:.5f}, "
						  f"AUC-PR: {average_metrics['AUC-PR'][class_idx]:.5f}, "
						  f"MCC: {average_metrics['MCC'][class_idx]:.5f}\n")
			log.write("\n")

		# After processing all folds, divide by total folds to get the final average
		for key in overall_metrics:
			overall_metrics[key] /= len(model_files)

		# Log the overall average metrics
		log.write("Overall Average Metrics Across All Models:\n")
		tqdm.write("Overall Average Metrics Across All Models:")
		for class_idx in range(Y_test[0].shape[1]):
			log.write(f"Class {class_idx + 1} - AUC-ROC: {overall_metrics['AUC-ROC'][class_idx]:.5f}, "
					f"AUC-PR: {overall_metrics['AUC-PR'][class_idx]:.5f}, "
					f"MCC: {overall_metrics['MCC'][class_idx]:.5f}\n")
			tqdm.write(f"Class {class_idx + 1} - AUC-ROC: {overall_metrics['AUC-ROC'][class_idx]:.5f}, "
					f"AUC-PR: {overall_metrics['AUC-PR'][class_idx]:.5f}, "
					f"MCC: {overall_metrics['MCC'][class_idx]:.5f}")
	return

import matplotlib.pyplot as plt  # Add for plotting

class CombinedModel(nn.Module):
	def __init__(self, cnn_output_dim, llm_output_dim, hidden_dim, nb_classes):
		super(CombinedModel, self).__init__()
		self.fc1 = nn.Linear(cnn_output_dim + llm_output_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, nb_classes)
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(0.2)

	def forward(self, cnn_output, llm_output):
		x = torch.cat((cnn_output, llm_output), dim=1)  # Combine CNN and LLM outputs
		x = self.fc1(x)
		self.activation(x)
		x = self.dropout(x)
		x = self.fc2(x)
		return x

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
	model_path = ["cnn_models_mean"]
	test_model(model_path[0], result_file="./cnn_models_mean/result.txt", k_fold=5)
	exit()

	# Load data for CNN
	if os.path.exists("cnn_embeddings.pth"):
		print("Loading saved CNN embeddings...")
		data = torch.load("cnn_embeddings.pth")
		X_train_cnn = data["X_train"]
		Y_train_cnn = data["Y_train"]
		X_test_cnn = data["X_test"]
		Y_test_cnn = data["Y_test"]
		X_val_cnn = data["X_val"]
		Y_val_cnn = data["Y_val"]
	else:
		X_train_cnn, X_test_cnn, X_val_cnn, Y_train_cnn, Y_test_cnn, Y_val_cnn = preprocess_data_onehot(
			left=4000,
			right=4000,
			k_fold=8
		)
		# Save the embeddings for future use
		torch.save({
			"X_train": X_train_cnn,
			"Y_train": Y_train_cnn,
			"X_test": X_test_cnn,
			"Y_test": Y_test_cnn,
			"X_val": X_val_cnn,
			"Y_val": Y_val_cnn
		}, "cnn_embeddings.pth")

	print(X_train_cnn[0].shape, Y_train_cnn[0].shape) #(12093, 4, 7998) (12093, 7)
	print(X_test_cnn[0].shape, Y_test_cnn[0].shape)
	print(X_val_cnn[0].shape, Y_val_cnn[0].shape)

	## Load data for LLM
	#if os.path.exists("llm_embeddings.pth"):
	#	print("Loading saved embeddings...")
	#	data = torch.load("llm_embeddings.pth")
	#	X_train_llm = data["X_train"]
	#	Y_train_llm = data["Y_train"]
	#	X_test_llm = data["X_test"]
	#	Y_test_llm = data["Y_test"]
	#	X_val_llm = data["X_val"]
	#	Y_val_llm = data["Y_val"]
	#else:
	#	print("Generating embeddings...")
	#	X_train_llm, X_test_llm, X_val_llm, Y_train_llm, Y_test_llm, Y_val_llm = preprocess_data_raw_with_embeddings(
	#		left=4000,
	#		right=4000
	#	)
	#	# Save the embeddings for future use
	#	torch.save({
	#		"X_train": X_train_llm,
	#		"Y_train": Y_train_llm,
	#		"X_test": X_test_llm,
	#		"Y_test": Y_test_llm,
	#		"X_val": X_val_llm,
	#		"Y_val": Y_val_llm
	#	}, "llm_embeddings.pth")

	#print(X_train_llm[0].shape, Y_train_llm[0].shape) 
	#print(X_test_llm[0].shape, Y_test_llm[0].shape) 
	#print(X_val_llm[0].shape, Y_val_llm[0].shape) 

	## Initialize CNN model
	cnn_layers = MultiscaleCNNLayers(
		in_channels=64,
		embedding_dim=4,  # For one-hot encoding
		pooling_size=8,
		pooling_stride=8,
		drop_rate_cnn=0.3,
		drop_rate_fc=0.3,
		length=7998, # Length of the input sequence
		nb_classes=7
	)
	
	### Initialize CNN model with mean pooling
	cnn_model_mean = MultiscaleCNNModel(
		layers=cnn_layers,
		num_classes=7,
		sequence_length=15,  # Adjust based on your sequence length
		aggregation_method="mean"
	).to(DEVICE)

	## Initialize CNN model with learnable weights
	#cnn_model_linear = MultiscaleCNNModel(
	#	layers=cnn_layers,
	#	num_classes=7,
	#	sequence_length=15,  # Adjust based on your sequence length
	#	aggregation_method="learnable_linear"
	#).to(DEVICE)

	#### Train CNN model with mean pooling
	train_model(
		model=cnn_model_mean,
		mname="CNN_Mean",
		X_train=X_train_cnn,
		Y_train=Y_train_cnn,
		X_test=X_test_cnn,
		Y_test=Y_test_cnn,
		X_val=X_val_cnn,
		Y_val=Y_val_cnn,
		batch_size=32,
		epochs=CNN_EPOCHS,
		save_path="./cnn_models_mean",
		log_file="cnn_training_log_mean.txt"
	)
	
	#train_model(
	#	model=cnn_model_linear,
	#	mname="CNN_Linear",
	#	X_train=X_train_cnn,
	#	Y_train=Y_train_cnn,
	#	X_test=X_test_cnn,
	#	Y_test=Y_test_cnn,
	#	X_val=X_val_cnn,
	#	Y_val=Y_val_cnn,
	#	batch_size=32,
	#	epochs=CNN_EPOCHS,
	#	save_path="./cnn_models_linear",
	#	log_file="cnn_training_log_linear.txt"
	#)

	#### Initialize LLM model
	###llm_model = LLMClassifier(
	###	output_dim=7
	###).to(DEVICE)

	####print(llm_model)
	#### Train LLM model
	###train_model(
	###	model=llm_model,
	###	mname="LLM",
	###	X_train=X_train_llm,
	###	Y_train=Y_train_llm,
	###	X_test=X_test_llm,
	###	Y_test=Y_test_llm,
	###	X_val=X_val_llm,
	###	Y_val=Y_val_llm,
	###	batch_size=32,
	###	epochs=50,
	###	lr=1e-5,
	###	save_path="./llm_models1",
	###	log_file="llm_training_log.txt"
	###)

	#### Initialize Ensemble model
	###ensemble_model = EnsembleModel(
	###	llm_model=llm_model,
	###	cnn_model=cnn_model,
	###	llm_output_dim7,  # Output dimension of LLM
	###	cnn_output_dim=7,    # Output dimension of CNN
	###	nb_classes=6
	###).to(DEVICE)

	### Train Ensemble model (implement a new function for this)
	### train_ensemble_model(...)

	### Initialize CNN and LLM models
	##cnn_layers = MultiscaleCNNLayers(
	##	in_channels=64,
	##	embedding_dim=4,  # For one-hot encoding
	##	pooling_size=8,
	##	pooling_stride=8,
	##	drop_rate_cnn=0.3,
	##	drop_rate_fc=0.3,
	##	length=7998,  # Length of the input sequence
	##	nb_classes=7
	##)

	##cnn_model = MultiscaleCNNModel(
	##	layers=cnn_layers,
	##	num_classes=7,
	##	sequence_length=15,
	##	aggregation_method="mean"
	##).to(DEVICE)

	##llm_model = LLMClassifier(output_dim=7).to(DEVICE)

	### Initialize combined model
	##combined_model = CombinedModel(
	##	cnn_output_dim=7,  # Output dimension of CNN
	##	llm_output_dim=7,  # Output dimension of LLM
	##	hidden_dim=128,
	##	nb_classes=7
	##).to(DEVICE)

	### Train CNN and LLM models separately (if not already trained)
	### ...existing code for training CNN and LLM...

	### Combine CNN and LLM outputs for prediction
	##train_model(
	##	model=combined_model,
	##	mname="Combined",
	##	X_train=(X_train_cnn, X_train_llm),
	##	Y_train=Y_train_cnn,  # Assuming same labels for CNN and LLM
	##	X_test=(X_test_cnn, X_test_llm),
	##	Y_test=Y_test_cnn,
	##	X_val=(X_val_cnn, X_val_llm),
	##	Y_val=Y_val_cnn,
	##	batch_size=32,	#	epochs=15,	#	save_path="./combined_models",	#	log_file="combined_training_log.txt"	#)	# Optimize thresholds and plot metrics for each fold	for fold in range(len(X_test_cnn)):		print(f"Optimizing thresholds for fold {fold + 1}")