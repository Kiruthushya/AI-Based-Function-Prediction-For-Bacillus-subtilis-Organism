import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import esm
from Bio import SeqIO
from sklearn.preprocessing import StandardScaler

device = "cpu"

# =====================================
# LOAD MODEL
# =====================================
class FusionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Load GO terms
go_terms = [line.strip() for line in open("go_terms.txt")]

model = FusionModel(646, len(go_terms))
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

# =====================================
# LOAD ESM MODEL
# =====================================
esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
esm_model.eval()
batch_converter = alphabet.get_batch_converter()

# =====================================
# LOAD FASTA
# =====================================
seqs = []
for record in SeqIO.parse("bacillus_subtilis_PY79.fasta", "fasta"):
    try:
        pid = record.id.split("|")[1]
    except:
        pid = record.id
    seqs.append((pid, str(record.seq)))

print("Total sequences:", len(seqs))

# =====================================
# GENERATE EMBEDDINGS
# =====================================
def get_embeddings(seqs):
    embeddings = []

    for i in range(0, len(seqs), 8):
        batch = seqs[i:i+8]
        labels, strs, tokens = batch_converter(batch)

        with torch.no_grad():
            results = esm_model(tokens, repr_layers=[30])

        reps = results["representations"][30]

        for j, (_, seq) in enumerate(batch):
            emb = reps[j, 1:len(seq)+1].mean(0)
            embeddings.append(emb.numpy())

    return np.vstack(embeddings)

print("Generating embeddings...")
X_seq = get_embeddings(seqs)

# =====================================
# LOAD STRUCTURE FEATURES
# =====================================
X_struct = np.load("structure_features.npy")
struct_ids = pd.read_csv("structure_ids.txt", header=None)[0].tolist()

struct_dict = {pid: X_struct[i] for i, pid in enumerate(struct_ids)}

# =====================================
# ALIGN
# =====================================
X_seq_new, X_struct_new, ids = [], [], []

for i, (pid, _) in enumerate(seqs):
    if pid in struct_dict:
        X_seq_new.append(X_seq[i])
        X_struct_new.append(struct_dict[pid])
        ids.append(pid)

X_seq_new = np.array(X_seq_new)
X_struct_new = np.array(X_struct_new)

print("After alignment:", len(ids))

# =====================================
# NORMALIZATION (RECREATE)
# =====================================
scaler_seq = StandardScaler()
scaler_struct = StandardScaler()

X_seq_new = scaler_seq.fit_transform(X_seq_new)
X_struct_new = scaler_struct.fit_transform(X_struct_new)

# =====================================
# FUSION
# =====================================
X = np.concatenate([X_seq_new, X_struct_new], axis=1)
X = torch.tensor(X, dtype=torch.float32)

# =====================================
# PREDICTION
# =====================================
with torch.no_grad():
    probs = torch.sigmoid(model(X)).numpy()

# =====================================
# TOP-K FUNCTION
# =====================================
def get_top_k(probs, k):
    return np.argsort(-probs, axis=1)[:, :k]

top3 = get_top_k(probs, 3)
top7 = get_top_k(probs, 7)

# =====================================
# SAVE RESULTS
# =====================================
rows = []

for i, pid in enumerate(ids):
    go3 = [go_terms[j] for j in top3[i]]
    go7 = [go_terms[j] for j in top7[i]]

    rows.append({
        "Protein_ID": pid,
        "Top3_GO": ";".join(go3),
        "Top7_GO": ";".join(go7)
    })

df_out = pd.DataFrame(rows)
df_out.to_csv("external_predictions.csv", index=False)

print("✅ Saved: external_predictions.csv")
