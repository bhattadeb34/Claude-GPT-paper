
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np
import os
import matplotlib.cm as cm
import textwrap
import sys
import string
import random
# Import the data_home variable from config.py
from config import data_home

path_to_dependent_functions=os.path.join(data_home, 'Claude-GPT-paper', 'dependent_functions')
sys.path.append(path_to_dependent_functions)
from loading_roar_colab_results import (
    distances_results_to_dataframe,
    calculate_distances_for_overall_results,
    calculate_latent_distance

)



def simplify_prompt(prompt, similar_prompts_criteria, different_prompts_criteria):
    """
    Simplifies the prompt based on predefined criteria.
    """
    # For each criterion in similar prompts, check if it exists in the prompt and simplify
    for criterion in similar_prompts_criteria + different_prompts_criteria:
        if criterion in prompt:
            # Find the start index of the criterion and slice the string from that point
            start_idx = prompt.find(criterion)
            return prompt[start_idx:]
    return prompt  # Return the original prompt if no criteria match


def create_prompt_mapping(distances_df):
    """
    Creates a mapping from simplified prompts to unique identifiers.
    """
    # Example classifications
    similar_prompts_criteria = ['similar molecules']
    different_prompts_criteria = ['completely different molecules']
    
    # Apply the simplification function to each prompt
    distances_df['Prompt Strategy'] = distances_df['Prompt Strategy'].apply(lambda prompt: simplify_prompt(prompt, similar_prompts_criteria, different_prompts_criteria))
    
    if 'Simplified Prompt' not in distances_df.columns:
        distances_df['Simplified Prompt'] = distances_df['Prompt Strategy'].apply(lambda x: x.split(".")[0].split(",")[0])

    # Separately collect similar and different prompts
    similar_prompts = [prompt for prompt in distances_df['Simplified Prompt'].unique() if any(criterion in prompt for criterion in similar_prompts_criteria)]
    different_prompts = [prompt for prompt in distances_df['Simplified Prompt'].unique() if any(criterion in prompt for criterion in different_prompts_criteria)]

    # Sort each list alphabetically
    similar_prompts_sorted = sorted(similar_prompts)
    different_prompts_sorted = sorted(different_prompts)

    # Combine the two lists, maintaining the categorical and alphabetical ordering
    sorted_prompts = similar_prompts_sorted + different_prompts_sorted

    # Assign identifiers to the sorted prompts in a deterministic manner
    prompt_mapping = {prompt: string.ascii_uppercase[i] for i, prompt in enumerate(sorted_prompts)}
    
    return prompt_mapping


def add_prompt_identifiers(dataframe, prompt_mapping):
    """
    Add a 'Prompt Identifier' column to the DataFrame based on the prompt mapping and sort by this identifier.
    """
    dataframe['Prompt Identifier'] = dataframe['Simplified Prompt'].map(prompt_mapping)

    # Sort the DataFrame based on 'Prompt Identifier'
    dataframe.sort_values(by='Prompt Identifier', inplace=True)

    return dataframe

def create_legend_handles(distances_df, box_plot):
    """
    Create custom legend handles with matching colors from the box plot, based on the unique prompts in distances_df.
    """
    # Extract unique prompt identifiers in the order they are plotted
    prompt_identifiers = distances_df['Prompt Identifier'].drop_duplicates().values
    
    # Extract colors used in the boxplot; assuming one color per 'Prompt Identifier' in their plotting order
    colors = [box_plot.get_patch_by_id(f"box{i}").get_facecolor() for i in range(len(prompt_identifiers))]
    
    # Map each prompt identifier to its color
    identifier_to_color = dict(zip(prompt_identifiers, colors))
    
    # Map prompt identifiers to their descriptions
    identifier_to_prompt = {identifier: distances_df[distances_df['Prompt Identifier'] == identifier]['Simplified Prompt'].iloc[0] for identifier in prompt_identifiers}
    
    # Create legend handles
    legend_handles = [mlines.Line2D([], [], color=identifier_to_color[id], marker='s', linestyle='None', markersize=10, label=f"{id}: {identifier_to_prompt[id]}")
                      for id in prompt_identifiers]
    
    return legend_handles


def plot_distances_with_custom_legend(plot_df, prompt_mapping, save_path,y_axis_label='Distances'):
    """
    Plot distances with a custom legend outside the plot, ensuring legend colors match plot colors and descriptions are accurate.
    Parameters:
    - plot_df: DataFrame prepared for plotting, with 'Prompt Identifier'.
    - prompt_mapping: Mapping of prompt identifiers to their descriptions.
    - save_path: Path to save the plot image.
    """
    # Define a consistent color palette
    num_prompts = plot_df['Prompt Identifier'].nunique()
    palette = sns.color_palette("Set1", n_colors=num_prompts)
    
    plt.figure(figsize=(20, 10))
    box_plot = sns.boxplot(data=plot_df, x='Prompt Identifier', y='Distance', palette=palette)
    
    plt.xlabel('Prompt Identifier')
    plt.ylabel('Distance')
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Distances for Generated SMILES by Prompt')
    
    # Assuming identifier_to_prompt maps identifiers to their full/simplified descriptions
    identifier_to_prompt = {v: k for k, v in prompt_mapping.items()}

    # Create legend handles with correct descriptions
    legend_handles = [mlines.Line2D([], [], color=palette[i], marker='s', linestyle='None', markersize=10,
                                    label=f"{identifier}: {identifier_to_prompt[identifier]}") for i, identifier in enumerate(plot_df['Prompt Identifier'].unique())]
    
    #plt.legend(handles=legend_handles, title='Prompt Descriptions', loc='best', bbox_to_anchor=(1, 1), fontsize='small')
    plt.ylabel(y_axis_label)
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches='tight')  # Save the figure
    plt.show()

def average_generated_vectors(df):
    np.random.seed(42)  # For NumPy operations
    random.seed(42)

    # Create a mapping from Prompt Identifier to Simplified Prompt
    prompt_to_simplified = df[['Prompt Identifier', 'Simplified Prompt']].drop_duplicates().set_index('Prompt Identifier')['Simplified Prompt'].to_dict()

    # Initialize a list to store the averaged results
    averaged_results = []

    # Group by parent_smile and Prompt Identifier to process each subgroup
    grouped = df.groupby(['parent_smile', 'Prompt Identifier'])

    for (parent_smile, prompt_identifier), group in grouped:
        # Calculate the averages and extract necessary values
        average_generated_latent_1 = group['generated_latent_1'].mean()
        average_generated_latent_2 = group['generated_latent_2'].mean()
        average_generated_latent_3 = group['generated_latent_3'].mean()
        parent_latent_1 = group['parent_latent_1'].iloc[0]
        parent_latent_2 = group['parent_latent_2'].iloc[0]
        parent_latent_3 = group['parent_latent_3'].iloc[0]

        # Append the results to the list
        averaged_results.append({
            'parent_smile': parent_smile,
            'Prompt Identifier': prompt_identifier,
            'parent_latent_1': parent_latent_1,
            'parent_latent_2': parent_latent_2,
            'parent_latent_3': parent_latent_3,
            'average_generated_latent_1': average_generated_latent_1,
            'average_generated_latent_2': average_generated_latent_2,
            'average_generated_latent_3': average_generated_latent_3
        })

    # Convert the list of dictionaries to a DataFrame
    averaged_df = pd.DataFrame(averaged_results)

    # Add 'Simplified Prompt' to averaged_df using the mapping
    averaged_df['Simplified Prompt'] = averaged_df['Prompt Identifier'].map(prompt_to_simplified)

    return averaged_df

def create_latent_embeddings_parent_generated(overall_results, smiles_latent_map):

    rows = []

    for parent_smile, results_list in overall_results.items():
        parent_latent = smiles_latent_map.get(parent_smile, {})
        if isinstance(parent_latent, pd.Series):
            parent_latent = parent_latent.to_dict()

        for result in results_list:
            prompt_strategy = result['prompt_strategy']  # Extract the prompt strategy

            for generated_smile in result['generated_smiles']:
                generated_latent = smiles_latent_map.get(generated_smile, {})
                if isinstance(generated_latent, pd.Series):
                    generated_latent = generated_latent.to_dict()

                row = {
                    'parent_smile': parent_smile,
                    'generated_smile': generated_smile,
                    'parent_latent_1': parent_latent.get('latent_1', None),
                    'parent_latent_2': parent_latent.get('latent_2', None),
                    'parent_latent_3': parent_latent.get('latent_3', None),
                    'generated_latent_1': generated_latent.get('latent_1', None),
                    'generated_latent_2': generated_latent.get('latent_2', None),
                    'generated_latent_3': generated_latent.get('latent_3', None),
                    'Prompt Strategy': prompt_strategy,  # Include the prompt strategy in the row
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    return df
def load_hexbin_data(hexbin_csv_filename):
    hexbin_df = pd.read_csv(hexbin_csv_filename)
    coords = hexbin_df[['latent_1', 'latent_2']].values
    return coords
def plot_each_prompt(ax, coords, df, prompt):
    ax.hexbin(coords[:, 0], coords[:, 1], gridsize=30, cmap='Blues', alpha=0.5)
    prompt_df = df[df['Prompt Identifier'] == prompt]

    for _, row in prompt_df.iterrows():
        ax.plot(row['parent_latent_1'], row['parent_latent_2'], 'o', mfc='none', mec='black', markersize=12, alpha=0.9)
        ax.annotate('', xy=(row['average_generated_latent_1'], row['average_generated_latent_2']), 
                    xytext=(row['parent_latent_1'], row['parent_latent_2']),
                    arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(prompt, fontsize=18)
    ax.set_xlabel('Latent 1', fontsize=16)
    ax.set_ylabel('Latent 2', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Make spines thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)

def adjust_layout_and_create_legend(fig, axes, num_prompts, df):
    for idx in range(num_prompts, len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the right margin to make space for the legend

    # Creating legend mapping
    prompt_to_simplified = df[['Prompt Identifier', 'Simplified Prompt']].drop_duplicates().set_index('Prompt Identifier')['Simplified Prompt'].to_dict()
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'{pid}: {sp}', markerfacecolor='none', markersize=5) for pid, sp in prompt_to_simplified.items()]
    #fig.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.05, 1), title="Prompt Descriptions", fontsize=16)

def plot_latent_space_averaged_by_prompt_grid(overall_results, smiles_latent_map, hexbin_csv_filename, save_path=None):
    latent_embeddings = create_latent_embeddings_parent_generated(overall_results, smiles_latent_map)
    prompt_mapping = create_prompt_mapping(latent_embeddings)
    latent_space_embeddings = add_prompt_identifiers(latent_embeddings, prompt_mapping)
    averaged_df = average_generated_vectors(latent_space_embeddings)
    coords = load_hexbin_data(hexbin_csv_filename)

    unique_prompts = sorted(averaged_df['Prompt Identifier'].unique())
    n_cols = 4  # Number of columns
    n_rows = int(np.ceil(len(unique_prompts) / n_cols))  # Calculate the number of rows needed
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows), sharex=True, sharey=True)  # Adjust figure size dynamically
    axes = axes.flatten()

    for prompt_idx, prompt in enumerate(unique_prompts):
        if prompt_idx < len(axes):  # Check to avoid indexing error
            plot_each_prompt(axes[prompt_idx], coords, averaged_df, prompt)

    adjust_layout_and_create_legend(fig, axes, len(unique_prompts), averaged_df)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()


"""

def plot_each_prompt(ax, coords, df, prompt):
    ax.hexbin(coords[:, 0], coords[:, 1], gridsize=30, cmap='Blues', alpha=0.5)
    prompt_df = df[df['Prompt Identifier'] == prompt]

    for _, row in prompt_df.iterrows():
        ax.plot(row['parent_latent_1'], row['parent_latent_2'], 'o', mfc='none', mec='black', markersize=12, alpha=0.9)
        ax.annotate('', xy=(row['average_generated_latent_1'], row['average_generated_latent_2']), 
                    xytext=(row['parent_latent_1'], row['parent_latent_2']),
                    arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(prompt, fontsize=18)
    ax.set_xlabel('Latent 1', fontsize=16)
    ax.set_ylabel('Latent 2', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
def adjust_layout_and_create_legend(fig, axes, num_prompts, df):
    for idx in range(num_prompts, len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the right margin to make space for the legend

    # Creating legend mapping
    prompt_to_simplified = df[['Prompt Identifier', 'Simplified Prompt']].drop_duplicates().set_index('Prompt Identifier')['Simplified Prompt'].to_dict()
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'{pid}: {sp}', markerfacecolor='none', markersize=5) for pid, sp in prompt_to_simplified.items()]
    fig.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.05, 1), title="Prompt Descriptions", fontsize=16)
def plot_latent_space_averaged_by_prompt_grid(overall_results, smiles_latent_map, hexbin_csv_filename, save_path=None):
    latent_embeddings = create_latent_embeddings_parent_generated(overall_results, smiles_latent_map)
    prompt_mapping = create_prompt_mapping(latent_embeddings)
    latent_space_embeddings = add_prompt_identifiers(latent_embeddings, prompt_mapping)
    averaged_df = average_generated_vectors(latent_space_embeddings)
    coords = load_hexbin_data(hexbin_csv_filename)

    unique_prompts = sorted(averaged_df['Prompt Identifier'].unique())
    grid_size = int(np.ceil(np.sqrt(len(unique_prompts))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20), sharex=True, sharey=True)
    axes = axes.flatten()

    for prompt_idx, prompt in enumerate(unique_prompts):
        plot_each_prompt(axes[prompt_idx], coords, averaged_df, prompt)

    adjust_layout_and_create_legend(fig, axes, len(unique_prompts), averaged_df)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()
"""


def plot_median_distances_by_prompt_bar_plot(overall_results, smiles_latent_map, save_path, dpi_value=300):
    # Generate distances DataFrame
    distances_results = calculate_distances_for_overall_results(overall_results, smiles_latent_map)
    distances_df = distances_results_to_dataframe(distances_results)
    # Create prompt mapping from distances DataFrame
    prompt_mapping = create_prompt_mapping(distances_df)
    
    # Add prompt identifiers to the DataFrame
    distances_df = add_prompt_identifiers(distances_df, prompt_mapping)
    
    median_distances = distances_df.groupby(['Parent SMILE', 'Prompt Identifier'])['Distance'].median().reset_index()
    
    # Create a color palette dictionary
    num_unique_prompts = median_distances['Prompt Identifier'].nunique()
    colormap = cm.get_cmap('Set1', num_unique_prompts)
    colors = [colormap(i) for i in range(num_unique_prompts)]
    unique_prompts = median_distances['Prompt Identifier'].unique()
    palette = dict(zip(unique_prompts, colors))
    
    # Adjusting the figure size here, consider increasing width and adjusting height
    plt.figure(figsize=(20, 12))  # Increased width for more space, adjust height as needed
    
    bar_plot = sns.barplot(data=median_distances, x='Parent SMILE', y='Distance', hue='Prompt Identifier', palette=palette)
    
    plt.xlabel('Parent SMILE',fontsize=16)
    plt.ylabel('Median Distance',fontsize=16)
    plt.xticks(rotation=90, ha='right',fontsize=16)  # Rotate labels more if necessary
    plt.yticks(fontsize=16) 
    plt.yscale("linear")
    
    #plt.legend(title='Prompt Strategy', loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
    
    # Adjusting layout, might need to tweak rect parameters based on the final figure size
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        plt.savefig(save_path, dpi=dpi_value, bbox_inches='tight')
    plt.show()




'''
def create_prompt_mapping(distances_df):
    """
    Create a mapping from simplified prompts to numerical identifiers.
    """
    # Ensure 'Simplified Prompt' column exists in distances_df
    if 'Simplified Prompt' not in distances_df.columns:
        distances_df['Simplified Prompt'] = distances_df['Prompt Strategy'].apply(lambda x: x.split(".")[0].split(",")[0])

    unique_prompts = distances_df['Simplified Prompt'].unique()
    return {prompt: f"Prompt {i+1}" for i, prompt in enumerate(unique_prompts)}


def add_prompt_identifiers(distances_df, prompt_mapping):
    """
    Add a 'Prompt Identifier' column to the DataFrame based on the prompt mapping.
    """
    plot_df = distances_df.copy()
    plot_df['Prompt Identifier'] = plot_df['Simplified Prompt'].map(prompt_mapping)
    return plot_df

'''

def plot_median_std_distances_with_custom_legend(overall_results, smiles_latent_map, save_path, plot_type='median', y_axis_label='Distances'):
    # Generate distances DataFrame
    distances_results = calculate_distances_for_overall_results(overall_results, smiles_latent_map)
    distances_df = distances_results_to_dataframe(distances_results)
    prompt_mapping = create_prompt_mapping(distances_df)
    distances_df = add_prompt_identifiers(distances_df, prompt_mapping)
    # Aggregate data based on plot_type
    if plot_type == 'median':
        aggregated_distances = distances_df.groupby(['Parent SMILE', 'Prompt Identifier'])['Distance'].median().reset_index()
        y_axis_label = 'Median Distance'
    elif plot_type == 'std_dev':
        aggregated_distances = distances_df.groupby(['Parent SMILE', 'Prompt Identifier'])['Distance'].std().reset_index()
        y_axis_label = 'Standard Deviation of Distance'
    else:
        raise ValueError("Invalid plot_type. Choose 'median' or 'std_dev'.")

    # Add 'Prompt Identifier' and sort
    #aggregated_distances['Prompt Number'] = aggregated_distances['Prompt Identifier'].apply(lambda x: int(x.split(' ')[-1]))
    #sorted_distances = aggregated_distances.sort_values('Prompt Number')

    # Plot
    num_prompts = aggregated_distances['Prompt Identifier'].nunique()
    palette = sns.color_palette("Set1", n_colors=num_prompts)
    
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=aggregated_distances, x='Prompt Identifier', y='Distance', palette=palette)
    
    plt.xlabel('Prompt Identifier',fontsize=16)
    plt.ylabel(y_axis_label,fontsize=16)
    plt.xticks(rotation=45, ha='right',fontsize=16)
    plt.yticks(fontsize=16)
    #plt.title(f'Distribution of {y_axis_label} for Generated SMILES by Prompt')
    
    # Create legend
    identifier_to_prompt = {v: k for k, v in prompt_mapping.items()}
    legend_handles = [mlines.Line2D([], [], color=palette[i], marker='s', linestyle='None', markersize=10, label=f"{id}: {identifier_to_prompt[id]}") for i, id in enumerate(aggregated_distances['Prompt Identifier'].unique())]
    
    plt.legend(handles=legend_handles, title='Prompt Descriptions', loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
def wrap_text(text, word_limit_per_line):
    """Wrap text to a new line after a specified width."""
    return '\n'.join(textwrap.wrap(text, width=word_limit_per_line, break_long_words=True, replace_whitespace=False))

def plot_prompt_mapping_table(prompt_mapping,save_path, word_limit_per_line=10):
    # Convert the prompt_mapping dictionary into a sorted list by the prompt numbers
    sorted_mapping = sorted(prompt_mapping.items(), key=lambda x: x[1])
    
    # Wrap descriptions based on word count limit
    wrapped_mapping = [(wrap_text(desc, word_limit_per_line * 10), num) for desc, num in sorted_mapping]
    

    # Calculate figure height dynamically based on the number of lines in the wrapped descriptions
    fig_height = sum(len(desc.split('\n')) * 0.5 for desc, _ in wrapped_mapping) + 0.5 # 0.5 is the base height for a single line

    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis('off')

    # Create the table
    colWidths = [0.1, 0.6]  # Column widths
    table = ax.table(cellText=[[num, desc] for desc, num in wrapped_mapping],
                     colLabels=['Identifier', 'Description'],
                     loc='center', cellLoc='center', colWidths=colWidths)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)  # Adjust table scaling

    plt.tight_layout()
    # Optionally, save the plot if a save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        plt.show()
def generate_prompt_table(overall_results, smiles_latent_map,word_limit_per_line=6, save_path=None):
    # Calculate distances for overall results
    distances_results = calculate_distances_for_overall_results(overall_results, smiles_latent_map)
    
    # Convert distances results to DataFrame
    distances_df = distances_results_to_dataframe(distances_results)
    
    # Create prompt mapping from distances DataFrame
    prompt_mapping = create_prompt_mapping(distances_df)
    
    # Plot prompt mapping table with specified word limit per line
    plot_prompt_mapping_table(prompt_mapping,save_path, word_limit_per_line)
    


def calculate_and_plot_distances_custom_legend(overall_results, smiles_latent_map, save_path, y_axis_label='Distances'):
    # Calculate distances for overall results
    distances_results = calculate_distances_for_overall_results(overall_results, smiles_latent_map)
    
    # Convert distances results to DataFrame
    distances_df = distances_results_to_dataframe(distances_results)
    
    # Create prompt mapping from distances DataFrame
    prompt_mapping = create_prompt_mapping(distances_df)
    
    # Add prompt identifiers to the DataFrame
    plot_df = add_prompt_identifiers(distances_df, prompt_mapping)
    
    # Plot distances with a custom legend
    plot_distances_with_custom_legend(plot_df, prompt_mapping, save_path, y_axis_label)



def box_plot_averaged_distances(overall_results, smiles_latent_map, save_path, y_axis_label='Distances'):
    not_sorted_latent_space_embeddings= create_latent_embeddings_parent_generated(overall_results, smiles_latent_map)
    prompt_mapping = create_prompt_mapping(not_sorted_latent_space_embeddings)
    latent_space_embeddings=add_prompt_identifiers(not_sorted_latent_space_embeddings, prompt_mapping)

    averaged_df = average_generated_vectors(latent_space_embeddings)
    
    # Initialize a list to store distances
    distance_records = []

    # Iterate through each generated molecule
    for index, row in latent_space_embeddings.iterrows():
        parent_smile = row['parent_smile']
        prompt = row['Prompt Identifier']
        


        # Find corresponding averaged vector for the same parent and prompt
        averaged_vector = averaged_df[(averaged_df['parent_smile'] == parent_smile) & (averaged_df['Prompt Identifier'] == prompt)].iloc[0]
        generated_vector = {'latent_1': row['generated_latent_1'], 'latent_2': row['generated_latent_2'], 'latent_3': row['generated_latent_3']}
        averaged_vector_dict = {'latent_1': averaged_vector['average_generated_latent_1'], 'latent_2': averaged_vector['average_generated_latent_2'], 'latent_3': averaged_vector['average_generated_latent_3']}
        
        # Calculate Euclidean distance
        distance = calculate_latent_distance(generated_vector, averaged_vector_dict)
        distance_records.append({'Prompt Identifier': prompt, 'Distance': distance})

    # Convert to DataFrame for easy plotting
    distances_plot_df = pd.DataFrame(distance_records)

    # Use the existing function for plotting
    plot_distances_with_custom_legend(distances_plot_df, prompt_mapping, save_path, y_axis_label)

"""
def plot_quantities_with_custom_legend(plot_df, prompt_mapping, save_path, y_axis_label='Distances', plot_type='box', y_axis_unit=None):
    plt.figure(figsize=(20, 10))
    
    if plot_type == 'box':
        sns.boxplot(data=plot_df, x='Prompt Identifier', y='Distance', palette="Set1")
    elif plot_type == 'violin':
        sns.violinplot(data=plot_df, x='Prompt Identifier', y='Distance', palette="Set1")
    else:
        raise ValueError("plot_type must be either 'box' or 'violin'.")

    # Construct y-axis label with units if provided
    ylabel = f"{y_axis_label} ({y_axis_unit})" if y_axis_unit else y_axis_label

    plt.xlabel('Prompt Identifier')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Distribution of {ylabel} by Prompt')

    prompt_descriptions = {value: key for key, value in prompt_mapping.items()}
    legend_handles = [
        mlines.Line2D([], [], color=sns.color_palette("Set1")[i], marker='s', linestyle='None', markersize=10, label=prompt_descriptions[prompt])
        for i, prompt in enumerate(plot_df['Prompt Identifier'].unique())
    ]
    
    plt.legend(handles=legend_handles, title='Prompt Descriptions', loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()
def plot_quantities_box_violin(overall_results, smiles_latent_map, quantity, save_path=None, plot_type='box', y_axis_unit=None):
    distances_results = calculate_distances_for_overall_results(overall_results, smiles_latent_map)
    distances_df = distances_results_to_dataframe(distances_results)
    prompt_mapping = create_prompt_mapping(distances_df)

    data = []
    for parent_smile, results_list in overall_results.items():
        for result in results_list:
            prompt = result['prompt_strategy']
            value = result.get(quantity)
            if value is not None:
                data.append({'Prompt': prompt, 'Value': value})

    data_df = pd.DataFrame(data)
    unique_prompts = data_df['Prompt'].unique()
    local_prompt_mapping = {prompt: f"Prompt {i+1}" for i, prompt in enumerate(unique_prompts)}
    data_df['Prompt Identifier'] = data_df['Prompt'].map(local_prompt_mapping)
    plot_df = data_df.rename(columns={'Value': 'Distance'})

    plot_quantities_with_custom_legend(plot_df, prompt_mapping, save_path, y_axis_label=quantity.capitalize(), plot_type=plot_type, y_axis_unit=y_axis_unit)
"""



def plot_quantities_with_custom_legend(plot_df, prompt_mapping, save_path, y_axis_label='Value'):

    # Setting up the plot
    num_prompts = plot_df['Prompt Identifier'].nunique()
    palette = sns.color_palette("Set1", n_colors=num_prompts)
    
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.boxplot(ax=ax, data=plot_df, x='Prompt Identifier', y='Distance', palette=palette)
    
    ax.set_xlabel('Prompt Identifier', fontsize=16)
    ax.set_ylabel(y_axis_label, fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    
    # Creating custom legend
    identifier_to_prompt = {v: k for k, v in prompt_mapping.items()}
    legend_handles = [
        mlines.Line2D([], [], color=palette[i], marker='s', linestyle='None', markersize=10, label=f"{identifier}: {identifier_to_prompt[identifier]}")
        for i, identifier in enumerate(sorted(prompt_mapping.values()))
    ]
    
    ax.legend(handles=legend_handles, title='Prompts', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.subplots_adjust(right=0.8)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()



def plot_quantities_box(overall_results, smiles_latent_map, quantity, save_path):
    def simplify_prompt(prompt):
        # For each criterion in similar prompts, check if it exists in the prompt and simplify
        for criterion in similar_prompts_criteria + different_prompts_criteria:
            if criterion in prompt:
                # Find the start index of the criterion and slice the string from that point
                start_idx = prompt.find(criterion)
                return prompt[start_idx:]
        return prompt  # Return the original prompt if no criteria match
    distances_results = calculate_distances_for_overall_results(overall_results, smiles_latent_map)
    distances_df = distances_results_to_dataframe(distances_results)
    prompt_mapping = create_prompt_mapping(distances_df)
    
    
    data = []
    for parent_smile, results_list in overall_results.items():
        for result in results_list:
            prompt = result['prompt_strategy']
            value = result.get(quantity)
            if value is not None:
                data.append({'Prompt Strategy': prompt, 'Value': value})


    data_df = pd.DataFrame(data)
    # Example classifications (adjust this logic based on your actual criteria for similarity)
    similar_prompts_criteria = ['similar molecules']  # Example criteria for similar prompts
    different_prompts_criteria = ['completely different molecules']  # Example criteria for different prompts
    data_df['Prompt Strategy']=data_df['Prompt Strategy'].apply(simplify_prompt)
    # Apply the simplification function to each prompt    
    if 'Simplified Prompt' not in data_df.columns:
        data_df['Simplified Prompt'] = data_df['Prompt Strategy'].apply(lambda x: x.split(".")[0].split(",")[0])
    data_df=add_prompt_identifiers(data_df, prompt_mapping)
    plot_df = data_df.rename(columns={'Value': 'Distance'})
    y_axis_label = quantity.replace('_', ' ').capitalize()
    plot_quantities_with_custom_legend(plot_df, prompt_mapping, save_path, y_axis_label)



def generate_and_save_plots(overall_results, smiles_latent_map, base_save_path=None):
    # Define the quantities to be plotted and their corresponding filenames
    quantities = {
        'validity_ratio': 'validity_ratio.pdf',
        'api_call_time': 'api_call_time.pdf',
        'chemical_diversity': 'chemical_diversity.pdf'
    }

    # Loop through the quantities dictionary
    for quantity, filename in quantities.items():
        print(f"Plotting {quantity}...")
        save_path = None  # Initialize save_path as None

        # Only construct save_path if base_save_path is provided
        if base_save_path:
            save_path = os.path.join(base_save_path, filename)

        # Plot the quantity with or without saving based on save_path
        plot_quantities_box(overall_results, smiles_latent_map, quantity, save_path)

        if save_path:
            print(f"Plot saved to {save_path}")
        else:
            print(f"Displayed plot for {quantity} without saving.")




def prepare_data_and_distances(overall_results, smiles_latent_map, quantities):
    def simplify_prompt(prompt):
        # For each criterion in similar prompts, check if it exists in the prompt and simplify
        for criterion in similar_prompts_criteria + different_prompts_criteria:
            if criterion in prompt:
                # Find the start index of the criterion and slice the string from that point
                start_idx = prompt.find(criterion)
                return prompt[start_idx:]
        return prompt  # Return the original prompt if no criteria match
    distances_results = calculate_distances_for_overall_results(overall_results, smiles_latent_map)
    distances_df = distances_results_to_dataframe(distances_results)
    prompt_mapping = create_prompt_mapping(distances_df)
    distance_plot_df = add_prompt_identifiers(distances_df, prompt_mapping)
    
    
    data = []
    for parent_smile, results_list in overall_results.items():
        for result in results_list:
            prompt = result['prompt_strategy']
            for quantity in quantities:  # Iterate over each quantity
                value = result.get(quantity)
                if value is not None:
                    # Append a dictionary with the prompt, quantity, and its value
                    data.append({'Prompt Strategy': prompt, 'Quantity': quantity, 'Value': value})
    

    data_df = pd.DataFrame(data)
    # Example classifications (adjust this logic based on your actual criteria for similarity)
    similar_prompts_criteria = ['similar molecules']  # Example criteria for similar prompts
    different_prompts_criteria = ['completely different molecules']  # Example criteria for different prompts
    data_df['Prompt Strategy']=data_df['Prompt Strategy'].apply(simplify_prompt)
    # Apply the simplification function to each prompt    
    if 'Simplified Prompt' not in data_df.columns:
        data_df['Simplified Prompt'] = data_df['Prompt Strategy'].apply(lambda x: x.split(".")[0].split(",")[0])
    data_df=add_prompt_identifiers(data_df, prompt_mapping)


    return data_df, distance_plot_df, prompt_mapping
def plot_stacked_plots(overall_results, smiles_latent_map, quantities, save_path):
    data_df, distance_plot_df, prompt_mapping = prepare_data_and_distances(overall_results, smiles_latent_map, quantities)
    
    num_plots = 1 + len(quantities)
    fig, axes = plt.subplots(num_plots, 1, figsize=(20, 5 * num_plots), sharex=True)
    
    if num_plots == 1:
        axes = [axes]  # Ensure axes is always a list

    palette = sns.color_palette("Set1", len(prompt_mapping))
    identifier_to_prompt = {v: k for k, v in prompt_mapping.items()}
    
    subplot_labels = ['({})'.format(chr(i)) for i in range(ord('a'), ord('z')+1)]
    subplot_labels = subplot_labels[:num_plots]  # Adjust the list to match the number of plots

    # Plot distances with its specific ylabel and label
    sns.boxplot(ax=axes[0], data=distance_plot_df, x='Prompt Identifier', y='Distance', palette=palette)
    axes[0].set_ylabel('Distances', fontsize=18)
    axes[0].text(-0.1, 1.1, subplot_labels[0], transform=axes[0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # Make spines thicker for distance plot
    for spine in axes[0].spines.values():
        spine.set_linewidth(1.5)

    # Plot each quantity with a proper ylabel and label
    for i, quantity in enumerate(quantities):
        filtered_data = data_df[data_df['Quantity'] == quantity]
        if not filtered_data.empty:
            sns.boxplot(ax=axes[i + 1], data=filtered_data, x='Prompt Identifier', y='Value', palette=palette)
            axes[i + 1].set_ylabel(quantity.replace('_', ' ').capitalize(), fontsize=18)
            # Add subplot label
            axes[i + 1].text(-0.1, 1.1, subplot_labels[i + 1], transform=axes[i + 1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

            # Make spines thicker for this quantity plot
            for spine in axes[i + 1].spines.values():
                spine.set_linewidth(1.5)

    # Set common x-axis labels
    for ax in axes:
        ax.set_xlabel('Prompt Identifier', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

    # Create a legend
    legend_handles = [
        mlines.Line2D([], [], color=palette[i], marker='s', linestyle='None', markersize=10, label=f"{identifier}: {identifier_to_prompt[identifier]}")
        for i, identifier in enumerate(sorted(prompt_mapping.values()))
    ]
    
    #fig.legend(handles=legend_handles, title='Prompts', loc='upper left', bbox_to_anchor=(0.75, 0.89), fontsize=15)
    plt.subplots_adjust(right=0.75)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
