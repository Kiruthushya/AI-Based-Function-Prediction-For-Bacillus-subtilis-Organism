import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# =========================
# 1️⃣ LOAD DATA
# =========================
X = np.load("combined_features.npy")        # features
structure_ids = np.load("structure_ids.npy")  # IDs used in fusion
labels_df = pd.read_csv("matrix_label_100.csv")

# =========================
# 2️⃣ ALIGN LABELS USING IDS ✅
# =========================
labels_df = labels_df.set_index("Entry")

aligned_labels = []

valid_indices = []

for i, pid in enumerate(structure_ids):
    if pid in labels_df.index:
        aligned_labels.append(labels_df.loc[pid].values)
        valid_indices.append(i)

# Keep only matched features
X = X[valid_indices]
y = np.array(aligned_labels)

print("Aligned Feature shape:", X.shape)
print("Aligned Label shape:", y.shape)

# =========================
# 3️⃣ SPLIT
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# =========================
# 4️⃣ NORMALIZATION
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# =========================
# 5️⃣ TO TENSOR
# =========================
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# =========================
# 6️⃣ MODEL
# =========================
class FusionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = FusionModel(X.shape[1], y.shape[1])

# =========================
# 7️⃣ LOSS
# =========================
pos_weight = (y_train.shape[0] - y_train.sum(dim=0)) / (y_train.sum(dim=0) + 1e-6)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

# =========================
# 8️⃣ TRAIN
# =========================
best_val = float("inf")

for epoch in range(50):

    model.train()
    loss = criterion(model(X_train), y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), y_val)

    print(f"Epoch {epoch+1} | Train {loss.item():.4f} | Val {val_loss.item():.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "final_best_model.pt")

# =========================
# 9️⃣ TEST
# =========================
model.load_state_dict(torch.load("final_best_model.pt"))

with torch.no_grad():
    probs = torch.sigmoid(model(X_test)).numpy()

roc = roc_auc_score(y_test.numpy(), probs, average="micro")

print("\n🔥 FIXED ROC-AUC:", roc)
