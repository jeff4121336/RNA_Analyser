import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import fm
from collections import OrderedDict

encoding_seq = OrderedDict([
	('A', [1, 0, 0, 0]),
	('C', [0, 1, 0, 0]),
	('G', [0, 0, 1, 0]),
	('T', [0, 0, 0, 1]),
	('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
	('-', [0, 0, 0, 0]),  # Pad
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Genedata:
	def __init__(self, id):
		self.id = id
		self.seq = None
		self.seqLeft = None
		self.seqRight = None
		self.length = None

	@staticmethod
	def create_gene(id, seq, left, right):    
		seq = seq.upper()
		seq = seq.replace("U", "T")
		seq = list(seq)
		for seqIndex in range(len(seq)):
			if seq[seqIndex] not in ["A", "C", "G", "T"]:
				seq[seqIndex] = "N"
		seq = "".join(seq)
		seqLength = len(seq)
		lineLeft = seq[: int(seqLength * left / (left + right))]
		lineRight = seq[int(seqLength * left / (left + right)) :]

		if len(lineLeft) >= left:
			lineLeft = lineLeft[:left]
		if len(lineRight) >= right:
			lineRight = lineRight[-right:]

		gene = Genedata(id)
		gene.seqLeft = lineLeft.rstrip()
		gene.seqRight = lineRight.rstrip()
		gene.length = seqLength
		return gene

	@classmethod
	def load_sequence(cls, dataset, left=1000, right=1000, predict=False):
		#
		# Load the dataset and create gene training and testing dataset
		# Reading files and creating gene object
		# Return the gene dataset (genes)
		#
		genes = []
		path = dataset
		print("Importing dataset {0}".format(dataset))

		with open(path, "r") as f:
			seqIndex = 0
			for line in f:
				if line[0] == ">":
					if seqIndex > 0:
						gene = cls.create_gene(id, seq, left, right)
						genes.append(gene)
					id = line.strip()
					seq = ""
				else:
					seq += line.strip()
				seqIndex += 1

			# last sequence
			gene = cls.create_gene(id, seq, left, right)
			genes.append(gene)

		genes = np.array(genes)

		if not predict:
			genes = genes[np.random.permutation(np.arange(len(genes)))]

		print("Total number of genes: {0}".format(genes.shape[0]))
		return genes

class AttentionMask(nn.Module):
	def __init__(
		self,
		hidden,
		da,
		r,
		returnAttention=False,
		attentionRegularizerWeight=0.001,
		normalize=False,
		attmod="smooth",
		sharpBeta=1,
	):
		super(AttentionMask, self).__init__()
		self.hidden = hidden # number of hidden units of input
		self.da = da # number of units in attention layer
		self.r = r # number of heads
		self.returnAttention = returnAttention
		self.attentionRegularizerWeight = attentionRegularizerWeight 
		self.normalize = normalize # normalize attention score
		self.attmod = attmod 
		self.sharpBeta = sharpBeta 

		self.W1 = nn.Parameter(torch.Tensor(hidden , da)) # weight for (hidden, da)
		self.W2 = nn.Parameter(torch.Tensor(da, r)) # weight for (da, r)
		
		# initialize weights
		nn.init.xavier_uniform_(self.W1) 
		nn.init.xavier_uniform_(self.W2)
		
		self.activation = torch.tanh

	def forward(self, H):
		H1 = H[:, :, :-1]  # (batch_size, n, hidden) <-> input
		attention_mask = H[:, :, -1]  # (batch_size, n, 1) <-> attention_mask

		H_t = self.activation(torch.matmul(H, self.W1))  # (batch_size, n, da)
		temp = torch.matmul(H_t, self.W2).permute(0, 2, 1)  # (batch_size, r, n)

		# mask add on temp for softmax, for padding regions
		# mask: (batch_size, n) -> (batch_size, r, n)
		mask = (1.0 - attention_mask.float()) * -10000.0
		temp += mask.unsqueeze(1).repeat(1, self.r, 1)
		# Clamp temp to prevent extreme values
		temp = torch.clamp(temp, min=-1e4, max=1e4)
		
		if self.attmod == "softmax":  # paper using smoothing method instead
			A = F.softmax(temp * self.sharpBeta, dim=-1)
		elif self.attmod == "smooth":  # equation (2)
			_E = torch.sigmoid(
				temp * self.sharpBeta
			)  # sigmoid all elements in Energy matrix
			sumE = _E.sum(dim=2, keepdim=True) + 1e-8
			A = _E / sumE

		if self.normalize:  # equation (3)
			length = attention_mask.float().sum(
				dim=1, keepdim=True
			) / attention_mask.size(1)
			lengthr = length.unsequeeze(1).repeat(1, self.r, 1)
			A = A * lengthr

		# equation (4)
		M = torch.bmm(A, H1)  # (batch_size, r, hidden) <-> Final context emebedding

		# pytorch doesn't have add_loss, need to add munually in training loop
		if self.attentionRegularizerWeight > 0.0:
			regularization_loss = self._attention_regularizer(A)
		else:
			regularization_loss = 0.0

		return M, regularization_loss, A # Output embedding, Regularization loss, Attention score

	def _attention_regularizer(self, attention):
		#
		# AAT - I
		# Equation (5)
		#
		batch_size = attention.size(0)
		identity = torch.eye(self.r, device=attention.device)  # (r, r)
		temp = (
			torch.bmm(attention, attention.permute(0, 2, 1)) - identity
		)  # (batch_size, r, r)
		penalty = self.attentionRegularizerWeight * temp.pow(2).sum() / batch_size
		return penalty

class GeneDataset(Dataset):
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		id, combined_sequence, rnafm_embedding, one_hot_encoding = self.data[idx]
		return id, combined_sequence, rnafm_embedding, one_hot_encoding

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
	

def preprocess_data(left=3999, right=3999, test_only=False, file=None):
	data = Genedata.load_sequence(
		dataset=file,
		left=left,
		right=right,
		predict=True,
	)

	X_test = []
	# Load RNA-FM model
	rna_fm = RNA_FM()
	
	for gene in data:
		# Get left and right sequences
		seq_left = gene.seqLeft
		seq_right = gene.seqRight

		# Pad sequences
		seq_left = seq_left.ljust(left, '-')
		seq_right = seq_right.rjust(right, '-')

		# Replace invalid characters with 'N'
		seq_left = ''.join([c if c in encoding_seq.keys() else 'N' for c in seq_left])
		seq_right = ''.join([c if c in encoding_seq.keys() else 'N' for c in seq_right])

		# Combine sequences
		combined_sequence = seq_left + seq_right

		# Generate RNA-FM embedding
		rnafm_embedding = rna_fm.embeddings([(gene.id, combined_sequence)])

		# Generate one-hot encoding
		one_hot_left = [encoding_seq[c] for c in seq_left]
		one_hot_right = [encoding_seq[c] for c in seq_right]
		one_hot_combined = np.array(one_hot_left + one_hot_right).T

		# Append the processed data
		X_test.append((gene.id, combined_sequence, rnafm_embedding, one_hot_combined))

	return X_test