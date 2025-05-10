import torch

def combine_embeddings(file1, file2, output_file):
	"""
	Combine two .pth embedding files into one.
	
	Args:
		file1 (str): Path to the first .pth file.
		file2 (str): Path to the second .pth file.
		output_file (str): Path to save the combined .pth file.
	"""
	# Load the two files with weights_only=False to suppress the warning
	data1 = torch.load(file1, weights_only=False)
	data2 = torch.load(file2, weights_only=False)

	# Ensure embeddings are tensors
	embeddings1 = torch.tensor(data1["embeddings"]) if not isinstance(data1["embeddings"], torch.Tensor) else data1["embeddings"]
	embeddings2 = torch.tensor(data2["embeddings"]) if not isinstance(data2["embeddings"], torch.Tensor) else data2["embeddings"]
	print(f"Loaded {file1} with {embeddings1.shape[0]} embeddings and {file2} with {embeddings2.shape[0]} embeddings.")
	
	# Combine embeddings, labels, and ids
	combined_embeddings = torch.cat((embeddings1, embeddings2), dim=0)
	combined_labels = data1["labels"] + data2["labels"]
	print(f"Loaded {file1} with {len(data1['labels'])} labels and {file2} with {len(data2['labels'])} labels.")

	combined_ids = data1["ids"] + data2["ids"]
	print(f"Loaded {file1} with {len(data1['ids'])} ids and {file2} with {len(data2['ids'])} ids.")

	print(f"Combined embeddings shape: {combined_embeddings.shape}")
	# Save the combined data
	torch.save({
		"embeddings": combined_embeddings,
		"labels": combined_labels,
		"ids": combined_ids
	}, output_file)

	print(f"Combined embeddings saved to {output_file}")


if __name__ == "__main__":
	# Example usage
	file1 = "./embedding_batches/embeddings_batch_0_11000.pth"
	file2 = "./embedding_batches/embeddings_batch_11000_17306.pth"
	output_file = "./embedding_batches/embeddings_batch_0_17306.pth"

	combine_embeddings(file1, file2, output_file)