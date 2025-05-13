from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import fm, json
import torch
from collections import OrderedDict
import tempfile
from dslayer import GeneDataset, preprocess_data
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from dslayer import MultiscaleCNNLayers, MultiscaleCNNModel, CombinedModel, RNA_FM
import os, shutil


app = Flask(__name__, static_url_path='/static')
modelSelected = None
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

num_channels = 30
kernel_size = 5
dropout_rate = 0.2
padding = 2
thresholds=[0.65, 0.94, 0.21, 0.56, 0.34, 0.29]

encoding_seq = OrderedDict([
	('A', [1, 0, 0, 0]),
	('C', [0, 1, 0, 0]),
	('G', [0, 0, 1, 0]),
	('T', [0, 0, 0, 1]),
	('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
	('-', [0, 0, 0, 0]),  # Pad
])

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
app = Flask(__name__)

# Directory containing your images

@app.route('/' , methods=['GET'])
def greeting():
	return render_template('index.html')

@app.route('/success', methods=['POST'])
def success():
	upload_method = request.form.get('upload_method')
	# Load the model
	cnn_model = MultiscaleCNNModel(
		layers=MultiscaleCNNLayers(
			in_channels=64,
			embedding_dim=4,
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
	cnn_model.load_state_dict(torch.load("./models/cnn.pth"), strict=False)

	model = CombinedModel(
		cnn_output_dim=7,
		llm_output_dim=640,
		hidden_dim=32,
		nb_classes=7
	).to(DEVICE)
	model.load_state_dict(torch.load("./models/llm.pth"), strict=False)

	model.eval()
	cnn_model.eval()
	
	if upload_method == 'sequence':
		rna_sequence = request.form.get('rna_sequence')
		seq = rna_sequence.upper().replace("U", "T")
		seq = list(seq)

		for seqIndex in range(len(seq)):
			if seq[seqIndex] not in ["A", "C", "G", "T"]:
				seq[seqIndex] = "N"
		
		seq = "".join(seq)
		seqLength = len(seq)
		lineLeft = seq[: int(seqLength * 4000 / 8000)]
		lineRight = seq[int(seqLength * 4000 / 8000) :]
		if len(lineLeft) >= 4000:
			lineLeft = lineLeft[:4000]
		else:
			lineLeft = lineLeft.ljust(4000, "-")
		if len(lineRight) >= 4000:
			lineRight = lineRight[-4000:]
		else:
			lineRight = lineRight.rjust(4000, "-")
		
		lineLeft = ''.join([c if c in encoding_seq.keys() else 'N' for c in lineLeft])
		lineRight = ''.join([c if c in encoding_seq.keys() else 'N' for c in lineRight])
		seq = lineLeft + lineRight

		one_hot_encoding = [encoding_seq[nucleotide] for nucleotide in seq]
		one_hot_encoding = torch.tensor(one_hot_encoding, dtype=torch.float32).to(device)
		one_hot_encoding = one_hot_encoding.unsqueeze(0)
		one_hot_encoding = one_hot_encoding.permute(0, 2, 1)  # Shape: [1, 4, 8000]
	
		rnafm = RNA_FM()
		data = [("CDS", seq)]
		rnafm_embeddings = rnafm.embeddings(data)

		cnn_embeddings, _ = cnn_model(one_hot_encoding)  
		outputs = model(cnn_embeddings, rnafm_embeddings)
		outputs = torch.sigmoid(outputs).squeeze(0).detach().cpu().numpy()
		outputs = outputs[:-1]
		output = []
		type = ["Nucleus", "Exosome", "Cytosol", "Ribosome", "Membrane", "Endoplastic reticulum"]
		for i in range(len(outputs)):
			typename = type[i]
			if outputs[i] >= thresholds[i]:
				output.append(f"{typename}: detected")
			else:
				output.append(f"{typename}: not detected")
			
		return render_template('index.html', 
						 result="The predictions have been successfully generated!", 
						 Sequence=seq,
						 onehot=one_hot_encoding.shape,
						 rnafm=rnafm_embeddings.shape, 
						 output=output)
	elif upload_method == 'file':
		fasta_file = request.files.get('fasta_file')

		# Save the uploaded file to a temporary location
		with tempfile.NamedTemporaryFile(delete=False) as temp_file:
			fasta_file.save(temp_file.name)
			temp_file_path = temp_file.name

		X_predict = preprocess_data(
			left=4000,
			right=4000,
			test_only=True,
			file=temp_file_path,
		)

		predict_dataset = GeneDataset(X_predict)
		predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False)
		predictions = []

		with torch.no_grad():
			for _, _, rnafm_embedding, one_hot_encoding in predict_loader:
				# Move data to the device
				rnafm_embedding = rnafm_embedding.to(DEVICE)
				one_hot_encoding = torch.tensor(one_hot_encoding, dtype=torch.float32).to(device)
				
				print(one_hot_encoding.shape)
				# Pass through the CNN model
				cnn_embeddings, _ = cnn_model(one_hot_encoding)
				outputs = model(cnn_embeddings, rnafm_embedding)
				print(outputs.shape)
				outputs = torch.sigmoid(outputs).detach().cpu().numpy()
				outputs = outputs[:,:-1]
				print(outputs.shape)
				output = []
				for output_row in outputs:
					output = (output_row >= thresholds).astype(int)
					predictions.append(output)

		# Write predictions to a file
		output_file_path = os.path.join("static", "predictions.txt")
		with open(output_file_path, "w") as output_file:
			output_file.write(f"Labels: Nucleus / Exosome / Cytosol / Ribosome / Membrane / Endoplastic reticulum \n")
			for i, (prediction, gene_data) in enumerate(zip(predictions, X_predict)):
				gene_id = gene_data[0]  # Extract the gene ID from the combined_sequence
				prediction_str = ", ".join([f"{p}" for p in prediction])
				output_file.write(f"{gene_id}: \n{prediction_str}\n")

		# Return the file to the user
		return render_template(
            'index.html',
			result="The predictions have been successfully generated!",
            download_url=url_for('static', filename='predictions.txt')
        )
	else: 
		# do nothing
		return '', 204
	

if __name__ == "__main__":
	app.run(port=5500, debug=True)
