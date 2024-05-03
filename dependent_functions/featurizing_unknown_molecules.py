import json
import pandas as pd
import joblib 
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import sys
import os

# Import the data_home variable from config.py
from config import data_home,fact_dropbox_path


sys.path.append(fact_dropbox_path)

from morgan_fingerprints import TransformMorganFingerprints
from morgan_fingerprints import WriteAllMorganFingerprints, TransformMorganFingerprints

def transform_new_molecules(smiles_list, radius, common_keys_path):
    """
    Generate and transform Morgan fingerprints for new molecules.

    Parameters:
    - smiles_list (list): List of SMILES strings.
    - radius (int): Radius for Morgan fingerprint generation.
    - common_keys_path (str): Path to the common keys JSON file.

    Returns:
    - DataFrame: Transformed fingerprints for the new molecules.
    """
    # Generate new Morgan fingerprints
    wf_mfp = WriteAllMorganFingerprints(radius=radius, extension='json')
    data_df = pd.DataFrame({'smiles': smiles_list})
    new_fingerprints = wf_mfp.transform(data_df)

    # Create an instance of the transformer
    transformer = TransformMorganFingerprints()

    # Set the common keys file path
    transformer.common_fp = common_keys_path

    # Transform the new fingerprints using the common keys
    transformed_new_fingerprints = transformer.transform(new_fingerprints)

    return transformed_new_fingerprints


def apply_pretrained_pca(df, pretrained_pca_path):
    """
    Applies a pre-trained PCA model to the DataFrame to generate latent space representations,
    returning a new DataFrame with the original 'smiles' column and new 'latent_X' columns
    for the PCA-transformed features.

    Parameters:
    - df: DataFrame containing the fragment features and 'smiles' column.
    - pretrained_pca_path: Path to the pre-trained PCA model (.lzma file).

    Returns:
    - DataFrame with the 'smiles' column and new 'latent_X' columns.
    """
    # Load the pre-trained model (scaler and PCA)
    try:
        scaler, pca = joblib.load(pretrained_pca_path)
    except ValueError:  # If only PCA is saved without a scaler
        pca = joblib.load(pretrained_pca_path)
        scaler = None

    # Select only fragment columns for PCA transformation
    fragment_cols = [col for col in df.columns if col.startswith('fragment_')]
    fragments_df = df[fragment_cols]
    
    # Standardize the fragment data if a scaler is present
    if scaler is not None:
        x = scaler.transform(fragments_df)
    else:
        x = fragments_df.values
    
    # Apply the PCA transform to the standardized fragment data
    latent_space = pca.transform(x)
    
    # Create a new DataFrame for the latent space representation
    latent_columns = [f'latent_{i+1}' for i in range(latent_space.shape[1])]
    latent_df = pd.DataFrame(latent_space, columns=latent_columns)
    
    # Add the 'smiles' column to the new DataFrame
    latent_df.insert(0, 'smiles', df['smiles'])

    return latent_df


def generate_transformed_latent_df(overall_results, common_keys_path, pretrained_pca_path, radius=2):
    aggregated_smiles = []

    # Collect SMILES from the keys of the dictionary and generated_smiles
    aggregated_smiles.extend(overall_results.keys())
    for entry in overall_results.values():
        for item in entry:
            aggregated_smiles.extend(item.get('generated_smiles', []))

    # Removing duplicates by converting the list to a set and back to a list
    aggregated_smiles = list(set(aggregated_smiles))

    # Assume transform_new_molecules and apply_pretrained_pca are previously defined functions
    transformed_data = transform_new_molecules(aggregated_smiles, radius=radius, common_keys_path=common_keys_path)
    transformed_latent_df = apply_pretrained_pca(transformed_data, pretrained_pca_path)
    
    return transformed_latent_df
def update_smiles_latent_map(transformed_latent_df, smiles_latent_map):
    # Iterate through the DataFrame to find smiles not in smiles_latent_map
    for idx, row in transformed_latent_df.iterrows():
        smile = row['smiles']
        if smile not in smiles_latent_map:
            # Add the new smile with its latent features to the smiles_latent_map
            smiles_latent_map[smile] = {
                'latent_1': row['latent_1'],
                'latent_2': row['latent_2'],
                'latent_3': row['latent_3']
            }
    return smiles_latent_map

