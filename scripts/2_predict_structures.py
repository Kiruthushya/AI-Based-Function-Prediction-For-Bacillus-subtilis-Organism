import torch
from transformers import EsmForProteinFolding, AutoTokenizer
from pathlib import Path
import re

# =========================
# INPUT / OUTPUT
# =========================
FASTA_PATH = "annotated_proteins.fasta"
OUTPUT_DIR = "structures"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# =========================
# LOAD MODEL
# =========================
print("Loading ESMFold model (HuggingFace)...")

model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

device = "cpu"
model = model.to(device)

print("Model loaded!")

# =========================
# READ FASTA (CORRECT WAY)
# =========================
def read_fasta(file):
    sequences = []
    name = None
    seq = []

    with open(file) as f:
        for line in f:
            line = line.strip()

            if line.startswith(">"):
                if name:
                    sequences.append((name, "".join(seq)))

                # clean protein name
                name = re.sub(r'[^a-zA-Z0-9_]', '_', line[1:].split()[0])
                seq = []

            else:
                seq.append(line)

        if name:
            sequences.append((name, "".join(seq)))

    return sequences


data = read_fasta(FASTA_PATH)

# =========================
# PREDICT STRUCTURES
# =========================
for name, seq in data:

    print(f"Processing {name}...")

    # Skip empty sequences (safety)
    if not seq:
        print(f"Skipping {name} (empty sequence)")
        continue

    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Convert to PDB
    pdb = model.output_to_pdb(outputs)[0]

    # Save file
    with open(f"{OUTPUT_DIR}/{name}.pdb", "w") as out:
        out.write(pdb)

print("✅ All structures generated!")
