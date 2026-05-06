import os
import numpy as np
from Bio.PDB import PDBParser

pdb_folder = "structures"
features = []
ids = []

parser = PDBParser(QUIET=True)

for file in os.listdir(pdb_folder):
    if file.endswith(".pdb"):
        path = os.path.join(pdb_folder, file)
        structure = parser.get_structure("protein", path)

        residues = []
        coords = []

        for model in structure:
            for chain in model:
                for res in chain:
                    if res.has_id("CA"):
                        residues.append(res.get_resname())
                        coords.append(res["CA"].get_coord())

        length = len(residues)

        if length == 0:
            continue

        coords = np.array(coords)

        # Structural features
        mean_dist = np.mean(np.linalg.norm(coords - coords.mean(axis=0), axis=1))
        std_dist = np.std(np.linalg.norm(coords - coords.mean(axis=0), axis=1))

        # Residue composition
        hydrophobic = sum(r in ["ALA","VAL","LEU","ILE","MET","PHE","TRP"] for r in residues) / length
        polar = sum(r in ["SER","THR","ASN","GLN"] for r in residues) / length
        charged = sum(r in ["LYS","ARG","ASP","GLU"] for r in residues) / length

        features.append([
            length,
            mean_dist,
            std_dist,
            hydrophobic,
            polar,
            charged
        ])

        ids.append(file.replace(".pdb",""))

features = np.array(features)

np.save("structure_features.npy", features)
np.save("structure_ids.npy", ids)

print("Structure features saved:", features.shape)
