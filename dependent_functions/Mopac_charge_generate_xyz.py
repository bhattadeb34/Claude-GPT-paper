import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def generate_and_select_conformer(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol) != 0:
        # Failed embedding
        return None
    AllChem.UFFOptimizeMolecule(mol)  # Optionally optimize the molecule
    return mol

def mol_to_xyz(mol, filename):
    xyz = ''
    num_atoms = mol.GetNumAtoms()
    xyz += str(num_atoms) + '\n\n'
    for atom in mol.GetAtoms():
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        xyz += f"{atom.GetSymbol()} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}\n"
    with open(filename, 'w') as f:
        f.write(xyz)
import os

def process_smiles(df, xyz_directory, smiles_mapping_file):
    # Ensure the main xyz_directory exists
    if not os.path.exists(xyz_directory):
        os.makedirs(xyz_directory)
    print(f"Saving XYZ files in directory: {xyz_directory}")

    # Create an 'output' subdirectory within the xyz_directory
    output_dir = os.path.join(xyz_directory, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Created 'output' directory for future use: {output_dir}")

    # Writing the SMILES mapping file in the main xyz_directory
    mapping_file_path = os.path.join(xyz_directory, smiles_mapping_file)
    with open(mapping_file_path, 'w') as smf:
        smf.write("Filename:SMILES\n")
        for index, row in df.iterrows():
            smiles = row['SMILES']
            mol = generate_and_select_conformer(smiles)
            if mol:
                filename = f"molecule_{index}.xyz"
                filepath = os.path.join(xyz_directory, filename)  # Save XYZ files in the main xyz_directory
                mol_to_xyz(mol, filepath)
                smf.write(f"{filename}:{smiles}\n")  # Write mapping to the SMILES mapping file
            else:
                print(f"Failed to process SMILES: {smiles}")
