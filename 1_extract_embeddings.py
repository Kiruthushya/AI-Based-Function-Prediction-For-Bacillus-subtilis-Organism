import esm
import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm

# load model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

model.eval()

embeddings = []

for record in tqdm(list(SeqIO.parse("annotated_proteins.fasta", "fasta"))):

    data = [(record.id, str(record.seq))]

    labels, strs, tokens = batch_converter(data)

    with torch.no_grad():
        results = model(tokens, repr_layers=[33])

    token_embeddings = results["representations"][33]

    seq_len = len(record.seq)

    embedding = token_embeddings[0, 1:seq_len+1].mean(0)

    embeddings.append(embedding.numpy())

embeddings = np.array(embeddings)

np.save("sequence_embeddings.npy", embeddings)

print("Saved sequence_embeddings.npy")
