import os
import json
import string
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import sys

# Import the data_home variable from config.py
from config import data_home

path_to_dependent_functions=os.path.join(data_home, 'Claude-GPT-paper', 'dependent_functions')
sys.path.append(path_to_dependent_functions)

from loading_roar_colab_results import map_smiles_to_latent_df
from Mopac_energy_charge_calculation import load_json_as_dict
from paper_plotting_results_notebook import calculate_distances_for_overall_results,distances_results_to_dataframe,create_prompt_mapping

def calculate_gasteiger_charges(mol):
    AllChem.ComputeGasteigerCharges(mol)
    return [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]

def calculate_molecule_metrics(mol):
    gasteiger_charges = calculate_gasteiger_charges(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    gasteiger_sum = sum(gasteiger_charges)
    gasteiger_avg = gasteiger_sum / len(gasteiger_charges)
    return {
        'gasteiger_charges': gasteiger_charges,
        'tpsa': tpsa,
        'gasteiger_sum': gasteiger_sum,
        'gasteiger_avg': gasteiger_avg
    }

def calculate_gasteiger_tpsa_metrics(overall_results):
    metrics_results = {}
    for reference_smiles, results_list in overall_results.items():
        reference_mol = Chem.MolFromSmiles(reference_smiles)
        if reference_mol:
            ref_metrics = calculate_molecule_metrics(reference_mol)
            metrics_results[reference_smiles] = {
                'reference': {**ref_metrics, 'smiles': reference_smiles},
                'generated': []
            }
            for result in results_list:
                for smiles in result['generated_smiles']:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        gen_metrics = calculate_molecule_metrics(mol)
                        metrics_results[reference_smiles]['generated'].append({**gen_metrics, 'smiles': smiles})
    return metrics_results

def map_simplify_prompt(prompt, criteria_list):
    for criterion in criteria_list:
        if criterion in prompt:
            return prompt[prompt.find(criterion):].split(".")[0].split(",")[0]
    return prompt

def calculate_metrics_for_prompts(overall_results):
    criteria_list = ['similar molecules', 'completely different molecules']
    prompt_metrics_results = defaultdict(list)
    for reference_smiles, results_list in overall_results.items():
        for result in results_list:
            simplified_prompt = map_simplify_prompt(result['prompt_strategy'], criteria_list)
            temp_overall_results = {reference_smiles: [result]}
            metrics_result = calculate_gasteiger_tpsa_metrics(temp_overall_results)
            prompt_metrics_results[simplified_prompt].append(metrics_result)
    return prompt_metrics_results

def calculate_deviation(ref_charges, gen_charges):
    ref_rms = np.sqrt(np.mean(np.square(ref_charges)))
    gen_rms = np.sqrt(np.mean(np.square(gen_charges)))
    return ref_rms - gen_rms

def calculate_gasteiger_deviations_by_prompt(overall_results):
    criteria_list = ['similar molecules', 'completely different molecules']
    deviations_by_prompt = defaultdict(list)
    for reference_smiles, results_list in overall_results.items():
        reference_mol = Chem.MolFromSmiles(reference_smiles)
        if reference_mol:
            ref_charges = np.array(calculate_gasteiger_charges(reference_mol))
            for result in results_list:
                simplified_prompt = map_simplify_prompt(result['prompt_strategy'], criteria_list)
                for smiles in result['generated_smiles']:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        gen_charges = np.array(calculate_gasteiger_charges(mol))
                        deviation = calculate_deviation(ref_charges, gen_charges)
                        deviations_by_prompt[simplified_prompt].append(deviation)
    return deviations_by_prompt

def calculate_gasteiger_deviations(overall_results):
    deviations_dict = {}
    for reference_smiles, results_list in overall_results.items():
        reference_mol = Chem.MolFromSmiles(reference_smiles)
        if reference_mol:
            ref_charges = np.array(calculate_gasteiger_charges(reference_mol))
            deviations_dict[reference_smiles] = []
            for result in results_list:
                for smiles in result['generated_smiles']:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        gen_charges = np.array(calculate_gasteiger_charges(mol))
                        deviation = calculate_deviation(ref_charges, gen_charges)
                        deviations_dict[reference_smiles].append(deviation)
    return deviations_dict

def update_deviation_dict_keys(prompt_mapping, deviation_dict_prompt):
    return {prompt_mapping.get(prompt, prompt): deviation_data for prompt, deviation_data in deviation_dict_prompt.items()}

def filter_deviation_dict(updated_deviation_dict, allowed_keys):
    return {key: value for key, value in updated_deviation_dict.items() if key in allowed_keys}

def filter_prompt_mapping(prompt_mapping, keys_to_keep):
    reversed_mapping = {value: key for key, value in prompt_mapping.items()}
    return {reversed_mapping[key]: key for key in keys_to_keep if key in reversed_mapping}

def load_json_as_dict(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def setup_paths(data_home, overall_results_filename):
    return {
        "common_keys_path": os.path.join(data_home, 'fact-dropbox/zinc/tranches/out/IdentifyCommonMFPKeys-WriteAllMorganFingerprints-ConcatCSV-2D-AK-AKEC-000.json'),
        "pretrained_pca_path": os.path.join(data_home, 'fact-dropbox/zinc/tranches/out/StructureEmbeddingMany-TransformMorganFingerprints-WriteAllMorganFingerprints-ConcatCSV-2D-AK-AKEC-009.lzma'),
        "overall_results_path": os.path.join(data_home, 'GPT_modification', 'out', overall_results_filename),
        "base_save_path": os.path.join(data_home, 'GPT_modification', 'out', 'Figures')
    }

def filter_by_prompt_strategy(overall_results, prompt_strategy):
    filtered_results = {}
    for parent_smiles, prompts in overall_results.items():
        filtered_prompts = [prompt for prompt in prompts if prompt['prompt_strategy'] == prompt_strategy]
        if filtered_prompts:
            filtered_results[parent_smiles] = filtered_prompts
    return filtered_results

def load_overall_results_smiles_latent_map(overall_results_path, radius, common_keys_path, pretrained_pca_path):
    overall_results = load_json_as_dict(overall_results_path)
    smiles_latent_map = map_smiles_to_latent_df(overall_results, radius, common_keys_path, pretrained_pca_path)
    return overall_results, smiles_latent_map

def process_chemical_data(data_home, overall_results_filename, radius, allowed_keys):
    paths = setup_paths(data_home, overall_results_filename)
    overall_results, smiles_latent_map = load_overall_results_smiles_latent_map(paths['overall_results_path'], radius,
                                                                                paths['common_keys_path'], paths['pretrained_pca_path'])
    distances_results = calculate_distances_for_overall_results(overall_results, smiles_latent_map)
    distances_df = distances_results_to_dataframe(distances_results)
    prompt_mapping = create_prompt_mapping(distances_df)
    criteria_list = ['similar molecules', 'completely different molecules']
    deviation_dict_prompt = {}
    list_of_prompts = list(set(prompt['prompt_strategy'] for results in overall_results.values() for prompt in results))
    for prompt in list_of_prompts:
        filtered_results = filter_by_prompt_strategy(overall_results, prompt)
        deviations = calculate_gasteiger_deviations(filtered_results)
        simplified_prompt = map_simplify_prompt(prompt, criteria_list)
        deviation_dict_prompt[simplified_prompt] = deviations
    filtered_prompt_mapping = filter_prompt_mapping(prompt_mapping, allowed_keys)
    updated_deviation_dict = update_deviation_dict_keys(prompt_mapping, deviation_dict_prompt)
    filtered_deviation_dict = filter_deviation_dict(updated_deviation_dict, allowed_keys)
    return filtered_deviation_dict, filtered_prompt_mapping, smiles_latent_map

def generate_EWG_prompt_mappings(EWG_deviations_dict, start_letter='I'):
    starting_index = string.ascii_uppercase.index(start_letter)
    sorted_EWG_prompts = sorted(EWG_deviations_dict.keys())
    return {prompt: string.ascii_uppercase[starting_index + i] for i, prompt in enumerate(sorted_EWG_prompts)}

def plot_gasteiger_charge_deviations_by_prompts(filtered_deviation_dict, EWG_deviations_dict, filtered_mapping, EWG_mapping, figsize=(16, 8), save_path=None):
    num_prompts = len(filtered_mapping) + len(EWG_mapping)
    primary_palette = sns.color_palette("Set1", n_colors=min(num_prompts, 9))
    palette = primary_palette + sns.color_palette("Set3", n_colors=num_prompts - 9) if num_prompts > 9 else primary_palette

    data_for_plotting = []
    for prompt, deviations_by_smiles in filtered_deviation_dict.items():
        for smiles, deviation_values in deviations_by_smiles.items():
            data_for_plotting.extend([{'Prompt': prompt, 'Deviation': deviation} for deviation in deviation_values])

    for prompt, deviations_list in EWG_deviations_dict.items():
        mapped_prompt = EWG_mapping[prompt]
        data_for_plotting.extend([{'Prompt': mapped_prompt, 'Deviation': deviation} for deviation in deviations_list])

    df_deviation = pd.DataFrame(data_for_plotting)

    plt.figure(figsize=figsize)
    plot_order = sorted(filtered_mapping.values()) + sorted(EWG_mapping.values())
    ax = sns.boxplot(x='Prompt', y='Deviation', data=df_deviation, palette=palette, order=plot_order)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Prompt Strategy', fontsize=14)
    plt.ylabel('RMSD of Gasteiger Charges\n(Parent - Generated)', fontsize=14)
    plt.title('Distribution of Deviations in Gasteiger Charges by Prompt Strategy', fontsize=16)

    legend_labels = {**filtered_mapping, **EWG_mapping}
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=textwrap.fill(f'{key}: {value}', width=50),
                                 markerfacecolor=palette[idx], markersize=10) for idx, (value, key) in enumerate(legend_labels.items())]
    legend = plt.legend(handles=legend_handles, title='Prompt Mapping', bbox_to_anchor=(1.05, 1), loc='upper left')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()