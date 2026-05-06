import numpy as np
import pandas as pd

# Load data
embeddings = np.load("sequence_embeddings.npy")
labels_df = pd.read_csv("matrix_label_100.csv")

structure_features = np.load("structure_features.npy")
structure_ids = np.load("structure_ids.npy")

# Map IDs to index
id_to_index = {id_: i for i, id_ in enumerate(labels_df["Entry"])}

combined_features = []
combined_labels = []

for i, pid in enumerate(structure_ids):
    if pid in id_to_index:
        idx = id_to_index[pid]

        emb = embeddings[idx]
        struct = structure_features[i]

        combined = np.concatenate([emb, struct])

        combined_features.append(combined)
        combined_labels.append(idx)

combined_features = np.array(combined_features)

np.save("combined_features.npy", combined_features)

print("Combined features shape:", combined_features.shape)
