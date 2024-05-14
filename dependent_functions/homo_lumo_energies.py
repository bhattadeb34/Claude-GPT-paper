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


sys.path.append('/noether/s0/dxb5775/GPT_modification/dependent_functions')
from pm7_mopac_deviations import map_simplify_mopac_prompt,load_mopac_overall_results_smiles_latent_map,setup_mopac_paths,filter_mopac_prompt_mapping,update_mopac_deviation_dict_keys,filter_mopac_deviation_dict,generate_mopac_EWG_prompt_mappings
from paper_plotting_results_notebook import calculate_distances_for_overall_results,distances_results_to_dataframe,create_prompt_mapping
def get_energy_folder_path(data_home, overall_results_filename):
    # Extract the base filename without the extension
    base_filename = os.path.splitext(overall_results_filename)[0]
    
    # Construct the folder name with 'xyz_' prefix
    xyz_folder_name = f"xyz_{base_filename}"
    
    # Combine with the data_home to form the full path
    xyz_folder_path = os.path.join(data_home, "GPT_modification", "pm7_charge_calculation", xyz_folder_name)
    
    return xyz_folder_path
def get_charge_folder_path(data_home, overall_results_filename):
    # Extract the base filename without the extension
    base_filename = os.path.splitext(overall_results_filename)[0]
    
    # Construct the folder name with 'xyz_' prefix
    charge_results_folder_name = f"charge_results_{base_filename}"
    
    # Combine with the data_home to form the full path
    charge_results_folder_path = os.path.join(data_home, "GPT_modification", "pm7_charge_calculation", charge_results_folder_name)
    
    return charge_results_folder_path
def calculate_homo_lumo_energy_differences(overall_results, parent_energies):
    energy_differences = defaultdict(lambda: defaultdict(list))

    for parent_smiles, results_list in overall_results.items():
        parent_homo = parent_energies.get(parent_smiles, {}).get('homo')
        parent_lumo = parent_energies.get(parent_smiles, {}).get('lumo')

        for result in results_list:
            generated_smiles_list = result.get('generated_smiles', [])

            for generated_smiles in generated_smiles_list:
                generated_homo = parent_energies.get(generated_smiles, {}).get('homo')
                generated_lumo = parent_energies.get(generated_smiles, {}).get('lumo')

                if parent_homo is not None and generated_homo is not None:
                    homo_difference = generated_homo - parent_homo  # Swap the order of subtraction
                    energy_differences[parent_smiles]['homo_differences'].append(homo_difference)

                if parent_lumo is not None and generated_lumo is not None:
                    lumo_difference = generated_lumo - parent_lumo  # Swap the order of subtraction
                    energy_differences[parent_smiles]['lumo_differences'].append(lumo_difference)

    return energy_differences
"""
#parent minus generated


def calculate_homo_lumo_energy_differences(overall_results, parent_energies):
    energy_differences = defaultdict(lambda: defaultdict(list))

    for parent_smiles, results_list in overall_results.items():
        parent_homo = parent_energies.get(parent_smiles, {}).get('homo')
        parent_lumo = parent_energies.get(parent_smiles, {}).get('lumo')

        for result in results_list:
            generated_smiles_list = result.get('generated_smiles', [])
            for generated_smiles in generated_smiles_list:
                generated_homo = parent_energies.get(generated_smiles, {}).get('homo')
                generated_lumo = parent_energies.get(generated_smiles, {}).get('lumo')

                if parent_homo is not None and generated_homo is not None:
                    homo_difference = parent_homo - generated_homo
                    energy_differences[parent_smiles]['homo_differences'].append(homo_difference)

                if parent_lumo is not None and generated_lumo is not None:
                    lumo_difference = parent_lumo - generated_lumo
                    energy_differences[parent_smiles]['lumo_differences'].append(lumo_difference)

    return energy_differences

def calculate_homo_lumo_energy_differences_by_prompt(overall_results, parent_energies, criteria_list):
    energy_differences_by_prompt = defaultdict(lambda: defaultdict(list))

    for parent_smiles, results_list in overall_results.items():
        parent_homo = parent_energies.get(parent_smiles, {}).get('homo')
        parent_lumo = parent_energies.get(parent_smiles, {}).get('lumo')

        for result in results_list:
            prompt = result['prompt_strategy']
            simplified_prompt = map_simplify_mopac_prompt(prompt, criteria_list)

            generated_smiles_list = result.get('generated_smiles', [])
            for generated_smiles in generated_smiles_list:
                generated_homo = parent_energies.get(generated_smiles, {}).get('homo')
                generated_lumo = parent_energies.get(generated_smiles, {}).get('lumo')

                if parent_homo is not None and generated_homo is not None:
                    homo_difference = parent_homo - generated_homo
                    energy_differences_by_prompt[simplified_prompt]['homo_differences'].append(homo_difference)

                if parent_lumo is not None and generated_lumo is not None:
                    lumo_difference = parent_lumo - generated_lumo
                    energy_differences_by_prompt[simplified_prompt]['lumo_differences'].append(lumo_difference)

    return energy_differences_by_prompt
"""
def calculate_homo_lumo_energy_differences_by_prompt(overall_results, parent_energies, criteria_list):
    energy_differences_by_prompt = defaultdict(lambda: defaultdict(list))

    for parent_smiles, results_list in overall_results.items():
        parent_homo = parent_energies.get(parent_smiles, {}).get('homo')
        parent_lumo = parent_energies.get(parent_smiles, {}).get('lumo')

        for result in results_list:
            prompt = result['prompt_strategy']
            simplified_prompt = map_simplify_mopac_prompt(prompt, criteria_list)
            generated_smiles_list = result.get('generated_smiles', [])

            for generated_smiles in generated_smiles_list:
                generated_homo = parent_energies.get(generated_smiles, {}).get('homo')
                generated_lumo = parent_energies.get(generated_smiles, {}).get('lumo')

                if parent_homo is not None and generated_homo is not None:
                    homo_difference = generated_homo - parent_homo  # Swap the order of subtraction
                    energy_differences_by_prompt[simplified_prompt]['homo_differences'].append(homo_difference)

                if parent_lumo is not None and generated_lumo is not None:
                    lumo_difference = generated_lumo - parent_lumo  # Swap the order of subtraction
                    energy_differences_by_prompt[simplified_prompt]['lumo_differences'].append(lumo_difference)

    return energy_differences_by_prompt

def process_homo_lumo_energy_data(data_home, overall_results_filename, homo_lumo_results, start_letter,radius=2, allowed_keys=None, ):
    criteria_list = ['similar molecules', 'completely different molecules']
    
    # Setup paths and load data
    paths = setup_mopac_paths(data_home, overall_results_filename)
    overall_results, smiles_latent_map = load_mopac_overall_results_smiles_latent_map(
        paths['overall_results_path'], radius, paths['common_keys_path'], paths['pretrained_pca_path']
    )
    
    # Calculate distances and create a prompt mapping
    distances_results = calculate_distances_for_overall_results(overall_results, smiles_latent_map)
    distances_df = distances_results_to_dataframe(distances_results)
    prompt_mapping = create_prompt_mapping(distances_df)
    
    # Calculate homo and lumo energy differences
    energy_differences = calculate_homo_lumo_energy_differences(overall_results, homo_lumo_results)
    energy_differences_by_prompt = calculate_homo_lumo_energy_differences_by_prompt(
        overall_results, homo_lumo_results, criteria_list
    )

    # Apply filtering if allowed_keys are provided
    if allowed_keys:
        print("Applying filtering for specified prompt keys.")
        filtered_prompt_mapping = filter_mopac_prompt_mapping(prompt_mapping, allowed_keys)
        updated_deviation_dict = update_mopac_deviation_dict_keys(prompt_mapping, energy_differences_by_prompt)
        filtered_deviation_dict = filter_mopac_deviation_dict(updated_deviation_dict, allowed_keys)
    else:
        print("Generating prompt mappings starting from letter:", start_letter)
        filtered_prompt_mapping = generate_mopac_EWG_prompt_mappings(energy_differences_by_prompt, start_letter=start_letter)
        filtered_deviation_dict = update_mopac_deviation_dict_keys(filtered_prompt_mapping, energy_differences_by_prompt)

    # Return necessary data for further analysis or plotting
    return filtered_deviation_dict, filtered_prompt_mapping, smiles_latent_map


def plot_homo_lumo_energy_differences_by_prompts(filtered_energy_diff_dict, EWG_energy_diff_dict, filtered_mapping, EWG_mapping, figsize=(16, 8), save_path=None):
    num_prompts = len(filtered_mapping) + len(EWG_mapping)
    primary_palette = sns.color_palette("Set1", n_colors=min(num_prompts, 9))
    palette = primary_palette + sns.color_palette("Set3", n_colors=num_prompts - 9) if num_prompts > 9 else primary_palette

    data_for_plotting = []

    # Collecting data for plotting
    for prompt, energy_diffs in filtered_energy_diff_dict.items():
        mapped_prompt = filtered_mapping.get(prompt, prompt)
        for key, value_list in energy_diffs.items():
            for value in value_list:
                data_for_plotting.append({
                    'Prompt': mapped_prompt, 
                    'Energy Difference': value, 
                    'Type': 'HOMO' if key == 'homo_differences' else 'LUMO'
                })

    for prompt, energy_diffs in EWG_energy_diff_dict.items():
        mapped_prompt = EWG_mapping.get(prompt, prompt)
        for key, value_list in energy_diffs.items():
            for value in value_list:
                data_for_plotting.append({
                    'Prompt': mapped_prompt, 
                    'Energy Difference': value, 
                    'Type': 'HOMO' if key == 'homo_differences' else 'LUMO'
                })

    df_energy_diff = pd.DataFrame(data_for_plotting)

    # Define plot order based on sorted mapping values
    plot_order = sorted(set(df_energy_diff['Prompt']))

    # Legend labels
    legend_labels = {**filtered_mapping, **EWG_mapping}

    # Plotting
    for type_key in ['HOMO', 'LUMO']:
        plt.figure(figsize=figsize)
        sns.boxplot(x='Prompt', y='Energy Difference', data=df_energy_diff[df_energy_diff['Type'] == type_key], palette=palette, order=plot_order)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Prompt Strategy', fontsize=14)
        plt.ylabel(f'{type_key} Energy Difference', fontsize=14)
        plt.title(f'Distribution of {type_key} Energy Differences by Prompt Strategy', fontsize=16)

        # Create and place legend for each plot
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                     label=textwrap.fill(f'{value}: {key}', width=50),
                                     markerfacecolor=palette[idx % len(palette)], markersize=10)
                          for idx, (key, value) in enumerate(legend_labels.items())]
        plt.legend(handles=legend_handles, title='Prompt Mapping', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_{type_key.lower()}.png", dpi=300, bbox_inches='tight')
        plt.show()


def plot_combined_homo_lumo_energy_differences(filtered_energy_diff_dict, EWG_energy_diff_dict, EDG_energy_diff_dict, filtered_mapping, EWG_mapping, EDG_mapping, figsize=(16, 8), condense_percentile=0.99, save_path=None):
    """
    Plots combined HOMO and LUMO energy differences by prompts.

    Parameters:
    - filtered_energy_diff_dict (dict): Energy differences for the A-D group prompts.
    - EWG_energy_diff_dict (dict): Energy differences for the EWG group prompts.
    - EDG_energy_diff_dict (dict): Energy differences for the EDG group prompts.
    - filtered_mapping (dict): Mapping of prompt descriptions to labels for A-D.
    - EWG_mapping (dict): Mapping of prompt descriptions to labels for EWG.
    - EDG_mapping (dict): Mapping of prompt descriptions to labels for EDG.
    - figsize (tuple): Dimensions of the figure to plot.
    - condense_percentile (float): Quantile to determine the y-axis limits to exclude extreme outliers. Default is 0.99.
    - save_path (str): If provided, where to save the plot image.

    Order of inputs is crucial for accurate plotting. Ensure data dictionaries and mappings follow the order: A-D, EWG, then EDG.
    """
    # Ensure input data structure integrity
    assert isinstance(filtered_energy_diff_dict, dict) and isinstance(EWG_energy_diff_dict, dict) and isinstance(EDG_energy_diff_dict, dict), "Energy difference inputs must be dictionaries"
    assert isinstance(filtered_mapping, dict) and isinstance(EWG_mapping, dict) and isinstance(EDG_mapping, dict), "Mapping inputs must be dictionaries"

    num_prompts = len(filtered_mapping) + len(EWG_mapping) + len(EDG_mapping)
    primary_palette = sns.color_palette("Set1", n_colors=min(num_prompts, 9))
    palette = primary_palette + sns.color_palette("Set3", n_colors=num_prompts - 9) if num_prompts > 9 else primary_palette

    data_for_plotting = []
    # Collect data for each type of prompts
    data_sources = [
        (filtered_energy_diff_dict, filtered_mapping, 'A-D'),
        (EWG_energy_diff_dict, EWG_mapping, 'EWG'),
        (EDG_energy_diff_dict, EDG_mapping, 'EDG')
    ]
    
    for energy_diff_dict, mapping, group_label in data_sources:
        for prompt, energy_diffs in energy_diff_dict.items():
            mapped_prompt = mapping.get(prompt, prompt)
            for key, value_list in energy_diffs.items():
                for value in value_list:
                    data_for_plotting.append({
                        'Prompt': mapped_prompt,
                        'Energy Difference': value,
                        'Type': 'HOMO' if key == 'homo_differences' else 'LUMO',
                        'Group': group_label
                    })

    df_energy_diff = pd.DataFrame(data_for_plotting)
    plot_order = sorted(set(df_energy_diff['Prompt']))

    # Determine y-limits based on specified percentile to exclude extreme outliers
    y_limit = df_energy_diff['Energy Difference'].quantile([1 - condense_percentile, condense_percentile]).abs().max()

    for type_key in ['HOMO']: #, 'LUMO']:
        plt.figure(figsize=figsize)
        ax = sns.boxplot(x='Prompt', y='Energy Difference', hue='Group', data=df_energy_diff[df_energy_diff['Type'] == type_key], palette=palette, order=plot_order)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlabel('Prompt Identifier', fontsize=24)
        #plt.ylabel(f'{type_key} Energy Difference (Parent-Generated) (EV)', fontsize=18)
        plt.ylabel(f'{type_key} Energy Difference\n(Generated-Parent) (eV)', fontsize=24)
        #plt.title(f'Combined Distribution of {type_key} Energy Differences by Prompt Strategy', fontsize=16)
        # Make spines thicker for distance plot
        for spine in ax.spines.values():
            spine.set_linewidth(2)  # Change this value to adjust the thickness
        # Draw a dashed line at y=0
        plt.axhline(0, color='gold', linestyle='dashed', linewidth=4)
        # Set y-limits to be symmetric around zero
        plt.ylim(-y_limit, y_limit)

        handles, labels = plt.gca().get_legend_handles_labels()
        new_labels = [textwrap.fill(label, width=50) for label in labels]
        #plt.legend(handles, new_labels, title='Groups', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=24, title_fontsize=24)
        plt.legend(handles, new_labels, title='Groups', loc='best', fontsize=24, title_fontsize=24)
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_{type_key.lower()}.pdf", dpi=600, bbox_inches='tight')
        plt.show()

    # for type_key in ['LUMO']:
    #     plt.figure(figsize=figsize)
    #     ax = sns.boxplot(x='Prompt', y='Energy Difference', hue='Group', data=df_energy_diff[df_energy_diff['Type'] == type_key], palette=palette, order=plot_order)
    #     plt.xticks(fontsize=24)
    #     plt.yticks(fontsize=24)
    #     plt.xlabel('Prompt Identifier', fontsize=24)
    #     #plt.ylabel(f'{type_key} Energy Difference (Parent-Generated) (EV)', fontsize=18)
    #     plt.ylabel(f'{type_key} Energy Difference\n(Generated-Parent) (eV)', fontsize=24)
    #     #plt.title(f'Combined Distribution of {type_key} Energy Differences by Prompt Strategy', fontsize=16)
    #     # Make spines thicker for distance plot
    #     for spine in ax.spines.values():
    #         spine.set_linewidth(2)  # Change this value to adjust the thickness
    #     # Draw a dashed line at y=0
    #     plt.axhline(0, color='gold', linestyle='dashed', linewidth=4)
    #     # Set y-limits to be symmetric around zero
    #     plt.ylim(-y_limit, y_limit)
    #
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     new_labels = [textwrap.fill(label, width=50) for label in labels]
    #     #plt.legend(handles, new_labels, title='Groups', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=24, title_fontsize=24)
    #     plt.legend(handles, new_labels, title='Groups', loc='best', fontsize=24, title_fontsize=24)
    #     plt.tight_layout()
    #     if save_path:
    #         plt.savefig(f"{save_path}_{type_key.lower()}.pdf", dpi=600, bbox_inches='tight')
    #     plt.show()

