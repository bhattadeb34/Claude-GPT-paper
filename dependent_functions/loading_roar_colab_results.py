import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
import numpy as np
import os
import sys
import json
# Import the data_home variable from config.py
from config import data_home
path_to_dependent_functions=os.path.join(data_home, 'Claude-GPT-paper', 'dependent_functions')
sys.path.append(path_to_dependent_functions)
from featurizing_unknown_molecules import transform_new_molecules, apply_pretrained_pca



def setup_paths(data_home,overall_results_filename):
    paths = {
        "common_keys_path": os.path.join(data_home, 'fact-dropbox/zinc/tranches/out/IdentifyCommonMFPKeys-WriteAllMorganFingerprints-ConcatCSV-2D-AK-AKEC-000.json'),
        "pretrained_pca_path": os.path.join(data_home, 'fact-dropbox/zinc/tranches/out/StructureEmbeddingMany-TransformMorganFingerprints-WriteAllMorganFingerprints-ConcatCSV-2D-AK-AKEC-009.lzma'),
        "overall_results_path": os.path.join(data_home, 'Claude-GPT-paper', 'out', overall_results_filename),
        "base_save_path": os.path.join(data_home, 'Claude-GPT-paper', 'out', 'Figures')
    }
    return paths

def load_json_as_dict(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def gather_all_smiles(overall_results):
    """
    Collects all unique SMILES strings from overall_results, including parent and generated children.
    """
    all_smiles_set = set()
    for parent_smile, results_list in overall_results.items():
        all_smiles_set.add(parent_smile)  # Add parent smile
        for result in results_list:
            generated_smiles = result['generated_smiles']
            all_smiles_set.update(generated_smiles)  # Add generated children smiles
    return list(all_smiles_set)

def get_latent_representations_for_all_smiles(smiles_list, radius, common_keys_path, pretrained_pca_path):
    """
    Transforms a list of SMILES strings into their latent representations.
    """
    # This function is assumed to be similar to transform_new_molecules and apply_pretrained_pca combined,
    # returning a DataFrame with latent representations for each SMILE in smiles_list.
    transformed_data = transform_new_molecules(smiles_list, radius=2, common_keys_path=common_keys_path)
    transformed_latent_df = apply_pretrained_pca(transformed_data, pretrained_pca_path)
    return transformed_latent_df

def map_smiles_to_latent_df(overall_results, radius, common_keys_path, pretrained_pca_path):
    """
    Maps each SMILES string to its latent representation.
    """
    all_smiles = gather_all_smiles(overall_results)
    latent_df = get_latent_representations_for_all_smiles(all_smiles, radius, common_keys_path, pretrained_pca_path)
    
    # Create a dictionary to map each SMILES to its latent representation excluding the SMILES string from the values
    smiles_latent_map = {}
    for _, row in latent_df.iterrows():
        smiles = row['smiles']
        # Exclude the 'smiles' column from the values and retain only the latent dimensions
        embeddings = row.drop('smiles')  # This creates a Series with just the latent dimensions
        smiles_latent_map[smiles] = embeddings
    
    return smiles_latent_map


def load_data(overall_results_path, radius, common_keys_path, pretrained_pca_path):
    overall_results = load_json_as_dict(overall_results_path)
    smiles_latent_map = map_smiles_to_latent_df(overall_results, radius, common_keys_path, pretrained_pca_path)
    return overall_results, smiles_latent_map


def calculate_latent_distance(smiles1_embeddings, smiles2_embeddings):
    """
    Calculate the Euclidean distance between the latent spaces of two SMILES strings.
    """
    # Ensure embeddings are numpy arrays for mathematical operations
    vec1 = np.array([smiles1_embeddings['latent_1'], smiles1_embeddings['latent_2'], smiles1_embeddings['latent_3']])
    vec2 = np.array([smiles2_embeddings['latent_1'], smiles2_embeddings['latent_2'], smiles2_embeddings['latent_3']])
    
    # Calculate Euclidean distance
    distance = np.linalg.norm(vec1 - vec2)
    return distance

def calculate_distances_for_overall_results(overall_results, smiles_latent_map):
    """
    Calculate distances between parent SMILES and generated SMILES for each prompt,
    including the generated SMILES in the results.
    """
    distances_results = {}  # Store distances and generated SMILES for each parent SMILE and prompt
    
    for parent_smile, results_list in overall_results.items():
        parent_embeddings = smiles_latent_map.get(parent_smile, {})
        distances_results[parent_smile] = []
        
        for result in results_list:
            prompt = result['prompt_strategy']
            distances_and_smiles_for_prompt = []
            
            for generated_smile in result['generated_smiles']:
                if generated_smile in smiles_latent_map:  # Ensure both SMILES have embeddings
                    generated_embeddings = smiles_latent_map[generated_smile]
                    distance = calculate_latent_distance(parent_embeddings, generated_embeddings)
                    distances_and_smiles_for_prompt.append((distance, generated_smile))
            
            # Store distances and generated SMILES for this prompt
            distances_results[parent_smile].append({
                'prompt': prompt,
                'distances_and_smiles': distances_and_smiles_for_prompt
            })
            
    return distances_results


def distances_results_to_dataframe(distances_results):
    """
    Convert the distances_results dictionary into a pandas DataFrame.
    """
    rows = []
    for parent_smile, prompts_list in distances_results.items():
        for prompt_info in prompts_list:
            prompt = prompt_info['prompt']
            for distance, generated_smile in prompt_info['distances_and_smiles']:
                row = {
                    'Parent SMILE': parent_smile,
                    'Generated SMILE': generated_smile,
                    'Distance': distance,
                    'Prompt Strategy': prompt
                }
                rows.append(row)
                
    return pd.DataFrame(rows)