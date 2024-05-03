import json
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw
import ipywidgets as widgets
from IPython.display import display, HTML
import string
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from IPython.display import display, HTML
import ipywidgets as widgets

def load_json_as_dict(filename):
    """
    Loads a JSON file into a Python dictionary.

    Parameters:
    - filename (str): The name of the JSON file to be loaded.

    Returns:
    - dict: The Python dictionary loaded from the JSON file.
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def calculate_rms_gasteiger_charges(mol):
    if mol is None:
        return 0
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
        rms = np.sqrt(np.mean(np.square(charges)))
        return round(rms, 3)
    except Exception as e:
        print(f"Error calculating charges: {e}")
        return 0



def draw_molecules(smiles_list):
    if not smiles_list:
        return "No molecules generated"

    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200), legends=[smiles for smiles in smiles_list])
    return img

def visualize_results(overall_results):
    # Visualization code remains largely the same, except no need to simplify prompts further
    parent_dropdown = widgets.Dropdown(options=list(overall_results.keys()), description='Parent Molecule:')
    output = widgets.Output()

    def on_parent_change(change):
        output.clear_output()
        with output:
            parent_mol = Chem.MolFromSmiles(change.new)
            parent_img = Draw.MolsToGridImage([parent_mol], molsPerRow=1, subImgSize=(200, 200), legends=[change.new])
            display(HTML('<h3>Parent Molecule</h3>'))
            display(parent_img)

            for result in overall_results[change.new]:
                display(HTML(f'<h3>{result["preprocessed_prompt"]}</h3>'))
                display(draw_molecules(result['generated_smiles']))

    parent_dropdown.observe(on_parent_change, names='value')
    display(parent_dropdown)
    display(output)

def preprocess_prompt(prompt):
    # Your existing logic to preprocess the prompt
    start_index = prompt.find("generate")
    end_index = prompt.find("Respond with")
    if start_index != -1 and end_index != -1:
        return prompt[start_index:end_index].strip()
    return prompt

def simplify_prompt(prompt):
    # Attempt to find and return a full prompt description based on a keyword match in `prompt_mapping`
    for simplified_prompt, full_description in prompt_mapping.items():
        if simplified_prompt in prompt:
            return full_description
    # Fallback: return the original prompt if no match is found
    return prompt

def process_and_visualize_results(overall_results):
    # Preprocess the prompts in overall_results directly without simplifying
    for parent_smiles, results in overall_results.items():
        for result in results:
            # Assuming preprocess_prompt is correctly defined
            # Directly assign the preprocessed prompt without simplification
            result['preprocessed_prompt'] = preprocess_prompt(result['prompt_strategy'])

    # Visualization doesn't need sorting based on prompt_mapping, proceed directly to visualization
    # Assuming visualize_results is correctly defined
    visualize_results(overall_results)


def calculate_rms_gasteiger_charges(mol):
    # Dummy implementation - replace with your actual RMS calculation
    AllChem.ComputeGasteigerCharges(mol)
    # Assuming you have a method to calculate and return RMS of Gasteiger charges
    return round(sum(float(atom.GetProp('_GasteigerCharge'))**2 for atom in mol.GetAtoms())**0.5, 2)

def draw_molecules_with_rms(smiles_list):
    mols = []
    rms_values = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:  # Check if the molecule was successfully created
            rms = calculate_rms_gasteiger_charges(mol)
            mols.append(mol)
            rms_values.append(f"RMS: {rms}")
        else:
            mols.append(None)
            rms_values.append("Invalid SMILES")
    return Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200), legends=rms_values)

def visualize_results_with_rms(overall_results):
    parent_dropdown = widgets.Dropdown(options=list(overall_results.keys()), description='Parent Molecule:')
    output = widgets.Output()

    def on_parent_change(change):
        output.clear_output()
        with output:
            parent_mol = Chem.MolFromSmiles(change.new)
            if parent_mol:  # Check if the molecule was successfully created
                parent_rms = calculate_rms_gasteiger_charges(parent_mol)
                parent_img = Draw.MolsToGridImage([parent_mol], molsPerRow=1, subImgSize=(200, 200), legends=[f"Parent (RMS: {parent_rms})"])
                display(HTML('<h3>Parent Molecule</h3>'))
                display(parent_img)

            for result in overall_results[change.new]:
                display(HTML(f'<h3>{result["preprocessed_prompt"]}</h3>'))  # Use the preprocessed prompt
                display(draw_molecules_with_rms(result['generated_smiles']))

    parent_dropdown.observe(on_parent_change, names='value')
    display(parent_dropdown)
    display(output)

