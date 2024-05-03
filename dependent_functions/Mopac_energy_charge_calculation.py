import os
import json
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
from ase import io
from ase.calculators.mopac import MOPAC
from tqdm import tqdm
import os

def extract_and_save_smiles(overall_results, save_path=None):
    """
    Extracts unique SMILES strings from overall results and saves them to a specified location.

    Parameters:
    - overall_results (dict): The dictionary containing overall results with parent and generated SMILES.
    - save_path (str): The path where the DataFrame of SMILES should be saved.

    Returns:
    - None
    """
    unique_smiles = set()

    # Extract parent SMILES (keys) and add to the set
    unique_smiles.update(overall_results.keys())

    # Extract generated SMILES and add to the set to ensure uniqueness
    for value_list in overall_results.values():
        for value in value_list:
            if 'generated_smiles' in value:
                unique_smiles.update(value['generated_smiles'])

    # Creating DataFrame from the list of unique SMILES
    df_smiles = pd.DataFrame(list(unique_smiles), columns=['SMILES'])

    # Saving DataFrame to the specified location
    if save_path:
        df_smiles.to_csv(save_path, index=False)
        print(f"SMILES saved to {save_path}")

    return df_smiles

def load_smiles_mapping(mapping_file):
    smiles_mapping = {}
    with open(mapping_file, 'r') as f:
        next(f)  # Skip the header line
        for line in f:
            if line.strip():
                file_path, smiles = line.strip().split(':')
                smiles_mapping[file_path] = smiles
    return smiles_mapping

def process_xyz_file(filename, output_dir, mopac_path):
    # Read the XYZ file into an Atoms object
    atoms = io.read(filename, format='xyz')
    
    # Determine file basename and calculation label
    basename = os.path.basename(filename)
    label_without_ext = os.path.splitext(basename)[0]
    
    # Set up the MOPAC calculation with a label that includes the output directory
    atoms.calc = MOPAC(label=os.path.join(output_dir, label_without_ext), task="1SCF", mopac_command=mopac_path)
    
    # Write the input file for MOPAC, it will be placed in the output directory
    atoms.calc.write_input(atoms)
    
    # Construct the MOPAC command to execute the calculation
    mopac_input_file = f"{label_without_ext}.mop"
    mopac_command = f"{mopac_path} {os.path.join(output_dir, mopac_input_file)}"
    os.system(mopac_command)
    
    # Read the output file to extract the charges
    output_file = f"{os.path.join(output_dir, label_without_ext)}.out"
    charges = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                if "NET ATOMIC CHARGES" in line:
                    next(f)  # Skip the header line
                    for charge_line in f:
                        if "DIPOLE" in charge_line:
                            break
                        parts = charge_line.split()
                        if len(parts) >= 3:
                            try:
                                charge = float(parts[2])
                                charges.append(charge)
                            except ValueError:
                                # Skip lines that don't contain valid charge values
                                continue
    else:
        print(f"Warning: Output file not found for {filename}")
    
    return filename, charges

def calculate_charges(xyz_files_directory, mopac_path, smiles_mapping_file):
    smiles_mapping = load_smiles_mapping(smiles_mapping_file)
    results = {}
    xyz_files = [f for f in os.listdir(xyz_files_directory) if f.endswith(".xyz")]
    for filename in tqdm(xyz_files, desc="Processing XYZ files"):
        full_path = os.path.join(xyz_files_directory, filename)
        if os.path.exists(full_path):
            output_dir = os.path.join(xyz_files_directory, "output")
            file_path, charges = process_xyz_file(full_path, output_dir, mopac_path)
            smiles = smiles_mapping.get(filename, 'N/A')
            results[smiles] = charges
        else:
            print(f"Warning: XYZ file not found: {full_path}")
    return results

def calculate_charges_multiple_runs(xyz_files_directory, mopac_path, smiles_mapping_file, num_runs=5):
    cumulative_charge_results = defaultdict(lambda: defaultdict(list))
    for _ in tqdm(range(num_runs), desc="Calculating charges across multiple runs"):
        current_run_results = calculate_charges(xyz_files_directory, mopac_path, smiles_mapping_file)
        for smiles, charges in current_run_results.items():
            for i, charge in enumerate(charges):
                cumulative_charge_results[smiles][i].append(charge)
    averaged_charge_results = {}
    for smiles, charge_dict in cumulative_charge_results.items():
        averaged_charges = [np.mean(charges) for charges in charge_dict.values()]
        averaged_charge_results[smiles] = averaged_charges
    return averaged_charge_results



def save_dict_as_json(data, filename):
    """
    Saves a dictionary as a JSON file, converting pandas Series to dictionaries if necessary.
    """
    data_to_save = {key: (value.to_dict() if isinstance(value, pd.Series) else value) for key, value in data.items()}
    with open(filename, 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print(f"Dictionary saved as {filename}")

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




def extract_homo_lumo_energies(filename, output_dir, mopac_path):
    # Read the XYZ file into an Atoms object
    atoms = io.read(filename, format='xyz')
    
    # Determine file basename and calculation label
    basename = os.path.basename(filename)
    label_without_ext = os.path.splitext(basename)[0]
    
    # Set up the MOPAC calculation with a label that includes the output directory
    atoms.calc = MOPAC(label=os.path.join(output_dir, label_without_ext), task="1SCF", mopac_command=mopac_path)
    
    # Write the input file for MOPAC, it will be placed in the output directory
    atoms.calc.write_input(atoms)
    
    # Construct the MOPAC command to execute the calculation
    mopac_input_file = f"{label_without_ext}.mop"
    mopac_command = f"{mopac_path} {os.path.join(output_dir, mopac_input_file)}"
    os.system(mopac_command)
    
    # Initialize variables for HOMO and LUMO energies
    homo_energy, lumo_energy = None, None
    
    # Read the output file to extract the HOMO and LUMO energies
    output_file = f"{os.path.join(output_dir, label_without_ext)}.out"
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                if "HOMO LUMO ENERGIES (EV)" in line:
                    parts = line.split("=")[1].strip().split()
                    if len(parts) >= 2:
                        try:
                            homo_energy = float(parts[0])
                            lumo_energy = float(parts[1])
                        except ValueError:
                            continue
                    break
                    
    if homo_energy is None or lumo_energy is None:
        print(f"Warning: HOMO/LUMO energies not found for {filename}")
    
    return filename, homo_energy, lumo_energy

def calculate_energies_multiple_runs(xyz_files_directory, mopac_path, smiles_mapping_file, num_runs=5):
    cumulative_energy_results = defaultdict(lambda: {'homo': [], 'lumo': []})
    
    xyz_dir = Path(xyz_files_directory)
    output_dir = xyz_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    for _ in tqdm(range(num_runs), desc="Calculating energies across multiple runs"):
        for xyz_file in xyz_dir.glob("*.xyz"):
            # Correctly capture the full path to the XYZ file
            full_xyz_path = xyz_dir / xyz_file
            filename, homo_energy, lumo_energy = extract_homo_lumo_energies(full_xyz_path, output_dir, mopac_path)
            smiles = load_smiles_mapping(smiles_mapping_file).get(xyz_file.name, 'N/A')
            if homo_energy is not None and lumo_energy is not None:
                cumulative_energy_results[smiles]['homo'].append(homo_energy)
                cumulative_energy_results[smiles]['lumo'].append(lumo_energy)

    averaged_energy_results = {}
    for smiles, energies in cumulative_energy_results.items():
        averaged_homo = np.mean(energies['homo']) if energies['homo'] else None
        averaged_lumo = np.mean(energies['lumo']) if energies['lumo'] else None
        averaged_energy_results[smiles] = {'homo': averaged_homo, 'lumo': averaged_lumo}
    
    return averaged_energy_results



def read_homo_lumo_energies(xyz_folder):
    output_folder = os.path.join(xyz_folder, "output")
    results = {}

    # Attempt to find the mapping file automatically in the output folder
    mapping_file = None
    for file in os.listdir(xyz_folder):
        if file.endswith(".txt"):
            mapping_file = os.path.join(xyz_folder, file)
            break

    if mapping_file is None:
        print(f"Error: No mapping file (.txt) found in {xyz_folder}")
        return results

    # Read the mapping file and create a dictionary mapping filenames to SMILES
    filename_to_smiles = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            if ':' in line:
                filename, smiles = line.strip().split(':')
                filename_to_smiles[filename] = smiles

    if os.path.exists(output_folder):
        for filename in os.listdir(output_folder):
            if filename.endswith(".out"):
                output_file = os.path.join(output_folder, filename)
                homo_energy, lumo_energy = None, None

                with open(output_file, 'r') as f:
                    for line in f:
                        if "HOMO LUMO ENERGIES (EV)" in line:
                            parts = line.split("=")[1].strip().split()
                            if len(parts) >= 2:
                                try:
                                    homo_energy = float(parts[0])
                                    lumo_energy = float(parts[1])
                                except ValueError:
                                    continue
                            break

                if homo_energy is None or lumo_energy is None:
                    print(f"Warning: HOMO/LUMO energies not found in {output_file}")
                    continue

                # Get the corresponding SMILES from the mapping dictionary
                filename_without_ext = os.path.splitext(os.path.basename(output_file))[0]
                smiles = filename_to_smiles.get(f"{filename_without_ext}.xyz")

                if smiles:
                    results[smiles] = {'homo': homo_energy, 'lumo': lumo_energy}
                else:
                    print(f"Warning: SMILES not found for {filename_without_ext}")
    else:
        print(f"Warning: Output folder '{output_folder}' does not exist.")

    return results