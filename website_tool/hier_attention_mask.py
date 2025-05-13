import torch
import torch.nn as nn
import torch.nn.functional as F


# Multi-Head Self Attention
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
