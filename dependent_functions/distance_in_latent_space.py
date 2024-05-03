import os
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import pandas as pd
import sys


# Import the data_home variable from config.py
from config import data_home
path_to_dependent_functions=os.path.join(data_home, 'Claude-GPT-paper', 'dependent_functions')
sys.path.append(path_to_dependent_functions)

from pm7_mopac_deviations import map_simplify_mopac_prompt,load_mopac_overall_results_smiles_latent_map,setup_mopac_paths,filter_mopac_prompt_mapping,update_mopac_deviation_dict_keys,filter_mopac_deviation_dict,generate_mopac_EWG_prompt_mappings
from paper_plotting_results_notebook import calculate_distances_for_overall_results,distances_results_to_dataframe,create_prompt_mapping

def calculate_latent_distance(parent_embeddings, generated_embeddings):
    """Calculate the Euclidean distance between the latent spaces of parent and generated SMILES."""
    # Ensure embeddings are numpy arrays for mathematical operations
    vec1 = np.array([parent_embeddings['latent_1'], parent_embeddings['latent_2'], parent_embeddings['latent_3']])
    vec2 = np.array([generated_embeddings['latent_1'], generated_embeddings['latent_2'], generated_embeddings['latent_3']])
    # Calculate Euclidean distance
    distance = np.linalg.norm(vec1 - vec2)
    return distance

def calculate_distances_by_prompt(overall_results, smiles_latent_map, criteria_list):
    distances_by_prompt = defaultdict(list)
    for parent_smiles, results_list in overall_results.items():
        parent_embeddings = smiles_latent_map.get(parent_smiles, {})
        for result in results_list:
            prompt = result['prompt_strategy']
            simplified_prompt = map_simplify_mopac_prompt(prompt, criteria_list)
            for generated_smiles in result['generated_smiles']:
                if generated_smiles in smiles_latent_map:
                    generated_embeddings = smiles_latent_map[generated_smiles]
                    distance = calculate_latent_distance(parent_embeddings, generated_embeddings)
                    distances_by_prompt[simplified_prompt].append(distance)
    return distances_by_prompt


def process_distance_data(data_home, overall_results_filename, smiles_latent_map, start_letter, radius=2, allowed_keys=None):
    criteria_list = ['similar molecules', 'completely different molecules']

    paths = setup_mopac_paths(data_home, overall_results_filename)
    overall_results = load_mopac_overall_results_smiles_latent_map(paths['overall_results_path'], radius,
                                                                   paths['common_keys_path'],
                                                                   paths['pretrained_pca_path'])[0]

    distances = calculate_distances_for_overall_results(overall_results, smiles_latent_map)
    distances_df = distances_results_to_dataframe(distances)
    prompt_mapping = create_prompt_mapping(distances_df)

    distances_by_prompt = calculate_distances_by_prompt(overall_results, smiles_latent_map, criteria_list)

    if allowed_keys:
        print("Applying filtering for specified prompt keys.")
        filtered_prompt_mapping = filter_mopac_prompt_mapping(prompt_mapping, allowed_keys)
        updated_deviation_dict = update_mopac_deviation_dict_keys(prompt_mapping, distances_by_prompt)
        filtered_deviation_dict = filter_mopac_deviation_dict(updated_deviation_dict, allowed_keys)
    else:
        print("Generating prompt mappings starting from letter:", start_letter)
        filtered_prompt_mapping = generate_mopac_EWG_prompt_mappings(distances_by_prompt, start_letter=start_letter)
        filtered_deviation_dict = update_mopac_deviation_dict_keys(filtered_prompt_mapping, distances_by_prompt)

    return filtered_deviation_dict, filtered_prompt_mapping


def plot_distances_by_prompts(filtered_distance_dict, EWG_distance_dict, EDG_distance_dict, filtered_mapping, EWG_mapping, EDG_mapping, figsize=(16, 8), save_path=None):
    num_prompts = len(filtered_mapping) + len(EWG_mapping) + len(EDG_mapping)
    primary_palette = sns.color_palette("Set1", n_colors=min(num_prompts, 9))
    palette = primary_palette + sns.color_palette("Set3", n_colors=num_prompts - 9) if num_prompts > 9 else primary_palette

    data_for_plotting = []
    data_sources = [
        (filtered_distance_dict, filtered_mapping, 'A-D'),
        (EWG_distance_dict, EWG_mapping, 'EWG'),
        (EDG_distance_dict, EDG_mapping, 'EDG')
    ]

    for distance_dict, mapping, group_label in data_sources:
        for prompt, distances in distance_dict.items():
            mapped_prompt = mapping.get(prompt, prompt)
            for distance in distances:
                data_for_plotting.append({
                    'Prompt': mapped_prompt,
                    'Distance': distance,
                    'Group': group_label
                })

    df_distances = pd.DataFrame(data_for_plotting)
    plot_order = sorted(set(df_distances['Prompt']))

    plt.figure(figsize=figsize)
    ax = sns.boxplot(x='Prompt', y='Distance', hue='Group', data=df_distances, palette=palette, order=plot_order)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Prompt Strategy', fontsize=18)
    plt.ylabel('Distance in Latent Space', fontsize=18)
    #plt.title('Distribution of Distances in Latent Space by Prompt Strategy', fontsize=16)

    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = [textwrap.fill(label, width=50) for label in labels]
    plt.legend(handles, new_labels, title='Groups', bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=16, title_fontsize=16)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

