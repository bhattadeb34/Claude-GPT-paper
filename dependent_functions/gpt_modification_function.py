import openai
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import json
import os
import time
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from openai import OpenAI
import time

def canonicalize_smiles(smiles):
    """Converts a SMILES string to its canonical form using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=True)
    return None
def load_api_key(file_path, keyword):
    """
    Assuming api keys are in a .txt file in the format:
    OPENAI_key="your_openai_key"
    CLAUDE_key="your_claude_key"
    and so on.
    Load an API key from a file based on a keyword.
    
    :param file_path: Path to the text file containing the API keys.
    :param keyword: Keyword to search for in the file (case-insensitive).
    :return: API key as a string if found, else None.
    """
    try:
        with open(file_path, 'r') as file:
            # Read all lines from the file
            lines = file.readlines()
        
        # Iterate through each line in the file
        for line in lines:
            # Check if the keyword is in the current line (case-insensitive)
            if keyword.lower() in line.lower():
                # Split the line on '=' and strip extra whitespace or quotes
                key_name, key_value = line.strip().split('=')
                # Return the API key, stripping surrounding quotes if present
                return key_value.strip().strip("'").strip('"')
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


'''
#previously implemented, before review
def is_valid_molecule(smiles):
    """Checks if a SMILES string represents a valid molecule."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None
'''

#suggestion from the reviewer to include other checks for validity, which was missing earlier
def is_valid_molecule(smiles):
    """
    Checks if a SMILES string represents a valid molecule.
    This includes converting the SMILES to a molecule object and performing sanitization.
    Returns the molecule object if valid, otherwise None.
    """
    try:
        # Convert SMILES to molecule object with sanitize=False
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None

        # Sanitize the molecule for validation
        Chem.SanitizeMol(mol)

        return mol
    except:
        return None



def build_prompt(smiles, n, prompt):
    return (f"Given the molecule with SMILES representation '{smiles}', generate {n} "
            f"molecules that are {prompt}. Respond with just the SMILES strings as elements of a Python list")



def calculate_tanimoto_similarity(smiles_list):
    # Generate fingerprints for each SMILES in the list, which are representations of molecules for similarity comparison
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=1024) for smiles in smiles_list]
    n = len(fps)
    similarity_sum = 0
    count = 0
    # Calculate pair-wise Tanimoto similarity between all pairs of fingerprints
    for i in range(n):
        for j in range(i + 1, n):
            similarity_sum += DataStructs.FingerprintSimilarity(fps[i], fps[j])
            count += 1
    # Average the similarities; Tanimoto similarity ranges from 0 (no similarity) to 1 (identical)
    average_similarity = similarity_sum / count if count > 0 else 0
    # Return diversity as 1 minus the average similarity; higher diversity indicates more dissimilarity among molecules
    return 1 - average_similarity


def make_api_call(prompt, key):
    start_time = time.time()  # Start time before API call
    client = OpenAI(api_key=key)
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a chemoinformatics expert that can generate new molecules. Please provide only the python formatted list of SMILES strings, like [SMILES1, SMILES2, SMILES3] without any additional explanations or text."},
            {"role": "user", "content": prompt}
        ]
    )
    response = completion.choices[0].message
    print(response)
    api_call_time = time.time() - start_time  # Measure API call time
    return response, api_call_time

def process_response(response_content, full_prompt, parent_smiles):
    start_time = time.time()
    raw_response_content = response_content.content if hasattr(response_content, 'content') else str(response_content)
    total_molecular_weights = 0
    unique_smiles = set()  # Use a set to store unique SMILES strings
    valid_smiles = []  # List to store valid SMILES for diversity calculation

    try:
        # Assuming response content is a direct JSON string of SMILES
        smiles_list = json.loads(raw_response_content)
    except json.JSONDecodeError:
        # Fallback for non-JSON content; attempting split assuming string representation
        smiles_list = raw_response_content.strip('[]').split(',')
        smiles_list = [smiles.strip(' "').strip() for smiles in smiles_list if smiles.strip()]

    total_generated = len(smiles_list)

    for smiles in smiles_list:
        if is_valid_molecule(smiles):
            canonical_smiles = canonicalize_smiles(smiles)
            if canonical_smiles and canonical_smiles != parent_smiles and canonical_smiles not in unique_smiles:
                unique_smiles.add(canonical_smiles)
                valid_smiles.append(canonical_smiles)  # Add to valid_smiles for diversity calculation
                mol = Chem.MolFromSmiles(canonical_smiles)
                total_molecular_weights += Descriptors.MolWt(mol)

    validity_ratio = len(unique_smiles) / total_generated if total_generated > 0 else 0
    generation_time = time.time() - start_time
    chemical_diversity = calculate_tanimoto_similarity(valid_smiles) if valid_smiles else 0
    average_molecular_weight = total_molecular_weights / len(unique_smiles) if unique_smiles else 0

    # Extracting the prompt strategy from the full prompt might need customization based on actual prompt format
    prompt_strategy = full_prompt.split(", generate")[1].strip() # Modify as needed based on your actual prompt strategies

    result = {
        'full_prompt': full_prompt,
        'number_of_total_generated_smiles_raw_responses': total_generated,
        'number_of_valid_canonicalized_smiles': len(unique_smiles),
        'prompt_strategy': prompt_strategy,
        'generated_smiles': list(unique_smiles),
        'total_raw_response_smiles': smiles_list,
        'raw_response': raw_response_content,
        'validity_ratio': validity_ratio,
        'post_processing_time': generation_time,
        'chemical_diversity': chemical_diversity,
        'average_molecular_weight': average_molecular_weight
    }

    return result


def prompt_strategies():
    # Returns a flat list of prompts
    return [
        "similar molecules with minimal structural changes to find similar but new candidates",
        "completely different molecules with significant structural changes to find new candidates",
        'similar molecules with slight variations on functional groups while maintaining the backbone structure',
        'completely different molecules that significantly vary in size and functional groups'
         ,'similar molecules by tweaking only the side chains',
         'completely different molecules by significantly altering the core structure and introducing completely new functional groups',
        'similar molecules by changing one or two atoms or bonds to produce closely related structures',
       'completely different molecules by changing multiple atoms or bonds'
    ]

def generate_smiles_for_prompts(smiles, n,key):
    """     
    API Call Time: Initiates when you make the call to openai.ChatCompletion.create(**chat_message) and 
    ends when you receive the response. This duration includes sending the data to OpenAI's servers, 
    waiting for the model to generate the response based on the input prompt, and receiving the response back.
    """    
    all_prompts = prompt_strategies()
    results = []
    for prompt in all_prompts:
        full_prompt = build_prompt(smiles, n, prompt)
        
        response, api_call_time = make_api_call(prompt,key)  # Unpack the tuple returned by make_api_call
        processed_response = process_response(response, full_prompt,smiles)
        processed_response['api_call_time'] = api_call_time  # Store API call time in results
        results.append(processed_response)
    return results


# modify generate_smiles_for_prompts function to have prompt_strategies as a function argument
def generate_smiles_for_custom_prompt(smiles, n, custom_prompt,key):
    """     
    API Call Time: Initiates when you make the call to openai.ChatCompletion.create(**chat_message) and 
    ends when you receive the response. This duration includes sending the data to OpenAI's servers, 
    waiting for the model to generate the response based on the input prompt, and receiving the response back.
    """    
    results = []
    for prompt in custom_prompt:
        full_prompt = build_prompt(smiles, n, prompt)
        
        response, api_call_time = make_api_call(prompt,key)  # Unpack the tuple returned by make_api_call
        processed_response = process_response(response, full_prompt,smiles)
        processed_response['api_call_time'] = api_call_time  # Store API call time in results
        results.append(processed_response)
    return results
# modify prompt_strategies function to return a single prompt with a specific strategy








#following is for claude
import backoff
import anthropic
def parse_response_to_list(response_text):
    smiles_list = eval(response_text)
    return smiles_list

@backoff.on_exception(backoff.expo, Exception, max_tries=5, max_time=120)
def make_api_call_claude(prompt,model,CLAUDE_API_KEY):
    start_time = time.time()
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        system="You are a chemoinformatics expert that can generate new molecules. Please provide only the python formatted list of SMILES strings, like [SMILES1, SMILES2, SMILES3] without any additional explanations or text.",
        messages=[
            {"role": "user", 
            "content": prompt}
        ]
    )
    print("Response content:", response.content)  # Debugging line to check the response content
    api_call_time = time.time() - start_time
    response_text = response.content[0].text
    parsed_smiles_list = parse_response_to_list(response_text)  # Use the function to parse response
    return parsed_smiles_list, api_call_time

def parse_response_claude(smiles_list, full_prompt):
    start_time = time.time()
    unique_smiles = set()  # Use a set to store unique SMILES strings
    valid_smiles = []  # List to store valid SMILES for diversity calculation
    total_molecular_weights = 0
    
    total_generated = len(smiles_list)
    print(f"total generated smiles {smiles_list}")
    
    for smiles in smiles_list:
        if is_valid_molecule(smiles):
            canonical_smiles = canonicalize_smiles(smiles)
            
            
            if canonical_smiles and canonical_smiles not in unique_smiles:
                unique_smiles.add(canonical_smiles)
                valid_smiles.append(canonical_smiles)
                
                
                mol = Chem.MolFromSmiles(canonical_smiles)
                total_molecular_weights += Descriptors.MolWt(mol)
    
    validity_ratio = len(unique_smiles) / total_generated if total_generated > 0 else 0
    generation_time = time.time() - start_time
    chemical_diversity = calculate_tanimoto_similarity(valid_smiles) if valid_smiles else 0
    average_molecular_weight = total_molecular_weights / len(unique_smiles) if unique_smiles else 0
    
    result = {
        'full_prompt': full_prompt,
        'number_of_total_generated_smiles_raw_responses': total_generated,
        'number_of_valid_canonicalized_smiles': len(unique_smiles),
        'prompt_strategy': full_prompt.split(", generate")[1].strip(),
        'generated_smiles': list(unique_smiles),
        'total_raw_response_smiles': smiles_list,
        'validity_ratio': validity_ratio,
        'post_processing_time': generation_time,
        'chemical_diversity': chemical_diversity,
        'average_molecular_weight': average_molecular_weight
    }
    
    return result
def generate_smiles_for_prompts_claude(smiles, n,all_prompts,CLAUDE_API_KEY, model="claude-3-sonnet-20240229"):
    results = []
    for prompt in all_prompts:
        full_prompt = build_prompt(smiles, n, prompt)
        response, api_call_time =make_api_call_claude(full_prompt,model,CLAUDE_API_KEY)
        processed_response = parse_response_claude(response, full_prompt)
        processed_response['api_call_time'] = api_call_time  # Store API call time in results
        results.append(processed_response)
    return results
