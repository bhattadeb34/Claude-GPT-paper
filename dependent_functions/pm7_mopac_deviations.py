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


def calculate_mopac_molecule_metrics(mol, smiles, charge_results):
    mopac_charges = charge_results.get(smiles, [])
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    mopac_sum = sum(mopac_charges)
    mopac_avg = mopac_sum / len(mopac_charges) if mopac_charges else 0
    return {
        'mopac_charges': mopac_charges,
        'tpsa': tpsa,
        'mopac_sum': mopac_sum,
        'mopac_avg': mopac_avg
    }

def calculate_mopac_tpsa_metrics(overall_results, charge_results):
    metrics_results = {}
    for reference_smiles, results_list in overall_results.items():
        reference_mol = Chem.MolFromSmiles(reference_smiles)
        if reference_mol:
            ref_metrics = calculate_mopac_molecule_metrics(reference_mol, reference_smiles, charge_results)
            metrics_results[reference_smiles] = {
                'reference': {**ref_metrics, 'smiles': reference_smiles},
                'generated': []
            }
            for result in results_list:
                for smiles in result['generated_smiles']:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        gen_metrics = calculate_mopac_molecule_metrics(mol, smiles, charge_results)
                        metrics_results[reference_smiles]['generated'].append({**gen_metrics, 'smiles': smiles})
    return metrics_results

def map_simplify_mopac_prompt(prompt, criteria_list):
    for criterion in criteria_list:
        if criterion in prompt:
            return prompt[prompt.find(criterion):].split(".")[0].split(",")[0]
    return prompt

def calculate_mopac_metrics_for_prompts(overall_results, charge_results):
    criteria_list = ['similar molecules', 'completely different molecules']
    prompt_metrics_results = defaultdict(list)
    for reference_smiles, results_list in overall_results.items():
        for result in results_list:
            simplified_prompt = map_simplify_mopac_prompt(result['prompt_strategy'], criteria_list)
            temp_overall_results = {reference_smiles: [result]}
            metrics_result = calculate_mopac_tpsa_metrics(temp_overall_results, charge_results)
            prompt_metrics_results[simplified_prompt].append(metrics_result)
    return prompt_metrics_results

def calculate_mopac_deviation(ref_charges, gen_charges):
    if len(ref_charges) == 0 or len(gen_charges) == 0:
        return 0
    ref_rms = np.sqrt(np.mean(np.square(ref_charges)))
    gen_rms = np.sqrt(np.mean(np.square(gen_charges)))
    return ref_rms - gen_rms

def calculate_mopac_deviations_by_prompt(overall_results, charge_results):
    criteria_list = ['similar molecules', 'completely different molecules']
    deviations_by_prompt = defaultdict(list)
    for reference_smiles, results_list in overall_results.items():
        ref_charges = np.array(charge_results.get(reference_smiles, []))
        for result in results_list:
            simplified_prompt = map_simplify_mopac_prompt(result['prompt_strategy'], criteria_list)
            for smiles in result['generated_smiles']:
                gen_charges = np.array(charge_results.get(smiles, []))
                deviation = calculate_mopac_deviation(ref_charges, gen_charges)
                deviations_by_prompt[simplified_prompt].append(deviation)
    return deviations_by_prompt

def calculate_mopac_deviations(overall_results, charge_results):
    deviations_dict = {}
    for reference_smiles, results_list in overall_results.items():
        ref_charges = np.array(charge_results.get(reference_smiles, []))
        deviations_dict[reference_smiles] = []
        for result in results_list:
            for smiles in result['generated_smiles']:
                gen_charges = np.array(charge_results.get(smiles, []))
                deviation = calculate_mopac_deviation(ref_charges, gen_charges)
                deviations_dict[reference_smiles].append(deviation)
    return deviations_dict

def update_mopac_deviation_dict_keys(prompt_mapping, deviation_dict_prompt):
    return {prompt_mapping.get(prompt, prompt): deviation_data for prompt, deviation_data in deviation_dict_prompt.items()}

def filter_mopac_deviation_dict(updated_deviation_dict, allowed_keys):
    return {key: value for key, value in updated_deviation_dict.items() if key in allowed_keys}

def filter_mopac_prompt_mapping(prompt_mapping, keys_to_keep):
    reversed_mapping = {value: key for key, value in prompt_mapping.items()}
    return {reversed_mapping[key]: key for key in keys_to_keep if key in reversed_mapping}

def load_mopac_json_as_dict(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def setup_mopac_paths(data_home, overall_results_filename):
    return {
        "common_keys_path": os.path.join(data_home,'Claude-GPT-paper', 'fact-dropbox/zinc/tranches/out/IdentifyCommonMFPKeys-WriteAllMorganFingerprints-ConcatCSV-2D-AK-AKEC-000.json'),
        "pretrained_pca_path": os.path.join(data_home,'Claude-GPT-paper', 'fact-dropbox/zinc/tranches/out/StructureEmbeddingMany-TransformMorganFingerprints-WriteAllMorganFingerprints-ConcatCSV-2D-AK-AKEC-009.lzma'),
        "overall_results_path": os.path.join(data_home, 'Claude-GPT-paper', 'out', overall_results_filename),
        "base_save_path": os.path.join(data_home, 'Claude-GPT-paper', 'out', 'Figures')
    }

def filter_mopac_results_by_prompt_strategy(overall_results, prompt_strategy):
    filtered_results = {}
    for parent_smiles, prompts in overall_results.items():
        filtered_prompts = [prompt for prompt in prompts if prompt['prompt_strategy'] == prompt_strategy]
        if filtered_prompts:
            filtered_results[parent_smiles] = filtered_prompts
    return filtered_results

def load_mopac_overall_results_smiles_latent_map(overall_results_path, radius, common_keys_path, pretrained_pca_path):
    overall_results = load_mopac_json_as_dict(overall_results_path)
    smiles_latent_map = map_smiles_to_latent_df(overall_results, radius, common_keys_path, pretrained_pca_path)
    return overall_results, smiles_latent_map

def process_mopac_chemical_data(data_home, overall_results_filename, radius, allowed_keys, charge_results):
    paths = setup_mopac_paths(data_home, overall_results_filename)
    overall_results, smiles_latent_map = load_mopac_overall_results_smiles_latent_map(paths['overall_results_path'], radius,
                                                                                paths['common_keys_path'], paths['pretrained_pca_path'])
    distances_results = calculate_distances_for_overall_results(overall_results, smiles_latent_map)
    distances_df = distances_results_to_dataframe(distances_results)
    prompt_mapping = create_prompt_mapping(distances_df)
    criteria_list = ['similar molecules', 'completely different molecules']
    deviation_dict_prompt = {}
    list_of_prompts = list(set(prompt['prompt_strategy'] for results in overall_results.values() for prompt in results))
    for prompt in list_of_prompts:
        filtered_results = filter_mopac_results_by_prompt_strategy(overall_results, prompt)
        deviations = calculate_mopac_deviations(filtered_results, charge_results)
        simplified_prompt = map_simplify_mopac_prompt(prompt, criteria_list)
        deviation_dict_prompt[simplified_prompt] = deviations
    filtered_prompt_mapping = filter_mopac_prompt_mapping(prompt_mapping, allowed_keys)
    updated_deviation_dict = update_mopac_deviation_dict_keys(prompt_mapping, deviation_dict_prompt)
    filtered_deviation_dict = filter_mopac_deviation_dict(updated_deviation_dict, allowed_keys)
    return filtered_deviation_dict, filtered_prompt_mapping, smiles_latent_map

def generate_mopac_EWG_prompt_mappings(EWG_deviations_dict, start_letter='I'):
    starting_index = string.ascii_uppercase.index(start_letter)
    sorted_EWG_prompts = sorted(EWG_deviations_dict.keys())
    return {prompt: string.ascii_uppercase[starting_index + i] for i, prompt in enumerate(sorted_EWG_prompts)}

def plot_mopac_charge_deviations_by_prompts(filtered_deviation_dict, EWG_deviations_dict, filtered_mapping, EWG_mapping, figsize=(16, 8), save_path=None):
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
    plt.ylabel('RMSD of MOPAC Charges\n(Parent - Generated)', fontsize=14)
    plt.title('Distribution of Deviations in MOPAC Charges by Prompt Strategy', fontsize=16)

    legend_labels = {**filtered_mapping, **EWG_mapping}
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=textwrap.fill(f'{key}: {value}', width=50),
                                 markerfacecolor=palette[idx], markersize=10) for idx, (value, key) in enumerate(legend_labels.items())]
    legend = plt.legend(handles=legend_handles, title='Prompt Mapping', bbox_to_anchor=(1.05, 1), loc='upper left')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
