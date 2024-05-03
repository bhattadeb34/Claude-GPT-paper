import json
from typing import List

import numpy as np
import rdkit.Chem
import rdkit.Chem.rdMolDescriptors


class MorganFeaturizer(object):
    """
    A class for featurizing molecules using Morgan fingerprints.

    Parameters:
    - thresh (int): The threshold for common keys identification.

    Methods:
    - compute_fingerprints(mols: List[rdkit.Chem.Mol]) -> List[dict]: Compute Morgan fingerprints for a list of molecules.
    - fit(mols: List[rdkit.Chem.Mol]) -> 'MorganFeaturizer': Identify common keys between molecules and store them for later use.
    - transform(mols: List[rdkit.Chem.Mol]) -> np.ndarray: Transform molecules into a matrix representation using common keys.
    - fit_transform(mols: List[rdkit.Chem.Mol]) -> np.ndarray: Fit the featurizer and transform molecules in a single step.
    - save(filepath: str): Save the featurizer's state to a JSON file.
    - load(filepath: str) -> None: Load the featurizer's state from a JSON file.
    """

    def __init__(self, thresh: int = 1, radius: int = 2):
        """
        Initialize the MorganFeaturizer.

        Parameters:
        - thresh (int): The threshold for common keys identification.
        """
        self.thresh = thresh
        self.radius = radius
        self.common_keys = None

    def compute_fingerprints(self, mols: List[rdkit.Chem.Mol]) -> List[dict]:
        """
        Compute Morgan fingerprints for a list of molecules.

        Parameters:
        - mols (list): List of RDKit molecules.

        Returns:
        - list: List of dictionaries representing Morgan fingerprints for each molecule.
        """
        fingerprints = []
        for mol in mols:
            fingerprints.append(rdkit.Chem.rdMolDescriptors.GetMorganFingerprint(mol, self.radius).GetNonzeroElements())
        return fingerprints

    def fit(self, mols: List[rdkit.Chem.Mol]) -> 'MorganFeaturizer':
        """
        Identify common keys between molecules and store them for later use.

        Parameters:
        - mols (list): List of RDKit molecules.

        Returns:
        - MorganFeaturizer: The fitted featurizer instance.
        """
        # get fingerprints
        all_fingerprints = self.compute_fingerprints(mols)

        # identify common keys between molecules
        all_keys = [key for fingerprint in all_fingerprints for key in fingerprint]
        all_keys = sorted(set(all_keys))

        key_counts = {k: 0 for k in all_keys}
        for i, fingerprint in enumerate(all_fingerprints):
            for key, val in fingerprint[0].items():
                key_counts[key] += 1

        # create a list of common keys
        key_lookup = {k: i for i, k in enumerate(all_keys)}

        full_mfp = np.zeros([len(all_fingerprints), len(all_keys)])
        for i, fingerprint in enumerate(all_fingerprints):
            for key, val in fingerprint.items():
                full_mfp[i, key_lookup[key]] = val

        common_idx = np.argwhere(full_mfp.sum(axis=0) > self.thresh).flatten()
        self.common_keys = {all_keys[j]: i for i, j in enumerate(common_idx)}

        return self

    def transform_mols(self, mols: List[rdkit.Chem.Mol]) -> np.ndarray:
        """
        Transform molecules into a matrix representation using common keys.

        Parameters:
        - mols (list): List of RDKit molecules.

        Returns:
        - numpy.ndarray: Matrix representation of molecules using common keys.
        """
        all_fingerprints = self.compute_fingerprints(mols)
        recon_mfp = np.zeros([len(all_fingerprints), len(self.common_keys)])
        for i, fingerprint in enumerate(all_fingerprints):
            for key, val in fingerprint.items():
                if key in self.common_keys:
                    recon_mfp[i, self.common_keys[key]] = val
        return recon_mfp

    def transform_counts(self, all_fingerprints) -> np.ndarray:
        recon_mfp = np.zeros([len(all_fingerprints), len(self.common_keys)])
        for i, fingerprint in enumerate(all_fingerprints):
            for key, val in fingerprint.items():
                if key in self.common_keys:
                    recon_mfp[i, self.common_keys[key]] = val
        return recon_mfp

    def transform(self, data):
        assert(type(data) is list)
        if type(data[0]) is rdkit.Chem.Mol:
            return self.transform_mols(data)
        elif type(data[0]):
            return self.transform_mols(data)
        else:
            raise ValueError('Expected list of rdkit.Chem.Mol or dict (representing precomputed fingerprints)')

    def fit_transform(self, mols: List[rdkit.Chem.Mol]) -> np.ndarray:
        """
        Fit the featurizer and transform molecules in a single step.

        Parameters:
        - mols (list): List of RDKit molecules.

        Returns:
        - numpy.ndarray: Matrix representation of molecules using common keys.
        """
        self.fit(mols)
        return self.transform(mols)

    def save(self, filepath: str) -> None:
        """
        Save the featurizer's state to a JSON file.

        Parameters:
        - filepath (str): Path to the output file.
        """
        state_dict = {'thresh': self.thresh,
                      'radius': self.radius,
                      'common_keys': self.common_keys}
        with open(filepath, 'w') as fid:
            json.dump(state_dict, fid)

    def load(self, filepath: str) -> None:
        """
        Load the featurizer's state from a JSON file.

        Parameters:
        - filepath (str): Path to the input file.
        """
        try:
            with open(filepath, 'r') as fid:
                state_dict = json.load(fid)

            self.thresh = state_dict['thresh']
            self.radius = state_dict['radius']
            self.common_keys = {int(k): v for k, v in state_dict['common_keys'].items()}

        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
            raise RuntimeError(f"Error loading featurizer state from {filepath}: {str(e)}")
