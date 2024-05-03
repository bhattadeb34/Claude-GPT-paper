# abstract base classes
from components.workflow import Workflow, ManyToOneWorkflow
# requirements for this instance
import joblib
import json
import numpy as np
import os
import pandas as pd
import rdkit as rdk
import rdkit.Chem
from modeling import MorganFeaturizer
from sklearn import decomposition, preprocessing


class PretrainedMorganFingerprints(Workflow):

    def read(self, filepath):
        suffix = filepath.split('.')[-1]
        if suffix == 'csv':
            return pd.read_csv(filepath, index_col=0)
        elif suffix == 'json':
            with open(filepath, 'r') as fid:
                data = json.load(fid)
            return data
        else:
            raise ValueError('Expected csv or json file.')

    def is_valid_file(self, filepath):
        # Try to read the file as a CSV
        # This is not OK for large files,
        # but it's the most precise method
        try:
            self.read(filepath)
            return True
        except:
            return False

    def write(self, data, filepath):
        data.to_csv(filepath)
        return filepath

    def transform(self, data):
        preproc = MorganFeaturizer()
        preproc.load(self.fragment_defs)
        smiles = []
        features = []
        for i in data.index:
            try:
                mol = rdk.Chem.MolFromSmiles(data.loc[i, 'smiles'])
            except:
                continue
            smiles.append(data.loc[i, 'smiles'])
            f = preproc.transform([mol])
            features.append(f)
        new_data = {'smiles': smiles}
        features = np.vstack(features).astype(int).T
        new_data.update({f'feature_{i}': f for i, f in enumerate(features)})
        new_df = pd.DataFrame(new_data)
        return new_df


class MorganFingerprints(Workflow):

    def read(self, filepath):
        return pd.read_csv(filepath)

    def is_valid_file(self, filepath):
        # Try to read the file as a CSV
        # This is not OK for large files,
        # but it's the most precise method
        try:
            self.read(filepath)
            return True
        except:
            return False

    def write(self, data, filepath):
        data.to_csv(filepath, index=False)
        return filepath

    def transform(self, data):
        if self.pretrained:
            preproc = MorganFeaturizer()
            preproc.load(self.fragment_defs)
        else:
            preproc = MorganFeaturizer(self.thresh, self.radius)
        smiles = []
        features = []
        for i in data.index:
            try:
                mol = rdk.Chem.MolFromSmiles(data.loc[i, 'smiles'])
            except:
                print(f'failed {i}')
                continue
            smiles.append(data.loc[i, 'smiles'])
            f = preproc.transform([mol])
            features.append(f)
        new_data = {'smiles': smiles}
        features = np.vstack(features).astype(int).T
        new_data.update({f'feature_{i}': f for i, f in enumerate(features)})
        new_df = pd.DataFrame(new_data)

        if not self.pretrained:
            if os.path.isfile(self.fragment_defs):
                raise FileExistsError(f'{self.fragment_defs} already exists!')
            preproc.save(self.fragment_defs)

        return new_df


class WriteAllMorganFingerprints(Workflow):

    def read(self, filepath):
        return pd.read_csv(filepath)

    def is_valid_file(self, filepath):
        # Try to read the file as a CSV
        # This is not OK for large files,
        # but it's the most precise method
        try:
            df = self.read(filepath)
            _ = df.loc[:, 'smiles']
            return True
        except:
            return False

    def write(self, data, filepath):
        if self.extension != 'json':
            raise ValueError('Writing to json format does not match specified file extension!')
        with open(filepath, 'w') as fid:
            json.dump(data, fid)
        return filepath

    def transform(self, data):
        mf = MorganFeaturizer(thresh=0, radius=self.radius)
        fragments = {}
        for i in data.index:
            if data.loc[i, 'smiles'] in fragments:
                print(f'skipping duplicate {i}: {data.loc[i, "smiles"]}')
                continue  # this is a duplicate entry
            try:
                mol = rdk.Chem.MolFromSmiles(data.loc[i, 'smiles'])
            except:
                print(f'failed {i}: {data.loc[i, "smiles"]}')
                continue
            fragments[data.loc[i, 'smiles']] = mf.compute_fingerprints([mol])[0]

        return fragments


class IdentifyCommonMFPKeys(ManyToOneWorkflow):
    # TODO: prevent this class from loading all the data files at once!
    # perhaps a new "Async" flavor of this class? it is only relevant with ManyToOne

    def read(self, filepath):
        with open(filepath, 'r') as fid:
            data = json.load(fid)
        return data

    def is_valid_file(self, filepath):
        # Try to read the file as a CSV
        # This is not OK for large files,
        # but it's the most precise method
        try:
            _ = self.read(filepath)
            return True
        except:
            return False

    def write(self, data, filepath):
        with open(filepath, 'w') as fid:
            json.dump(data, fid)
        return filepath

    def transform(self, data):
        # identify common keys
        all_fingerprints = {}
        for d in data:
            all_fingerprints.update(d)

        print(f'Merging {len(all_fingerprints)} fingerprints...', end='')

        all_keys = [key for fingerprint in all_fingerprints.values() for key in fingerprint]
        all_keys = sorted(set(all_keys))

        key_counts = {k: 0 for k in all_keys}
        for i, fingerprint in enumerate(all_fingerprints.values()):
            for key, val in fingerprint.items():
                key_counts[key] += 1

        if self.thresh < 1:
            thresh_int = self.thresh * len(all_fingerprints)
        else:
            thresh_int = int(self.thresh)

        common_keys = []
        for key in all_keys:
            if key_counts[key] > thresh_int:
                common_keys.append( key )

        print(f'chose {len(common_keys)} common keys...')

        return common_keys


class TransformMorganFingerprints(Workflow):

    def read(self, filepath):
        with open(filepath, 'r') as fid:
            data = json.load(fid)
        return data

    def is_valid_file(self, filepath):
        # Try to read the file as a CSV
        # This is not OK for large files,
        # but it's the most precise method
        try:
            _ = self.read(filepath)
            return True
        except:
            return False

    def write(self, data, filepath):
        data.to_csv(filepath, index=None)
        return filepath

    def transform(self, data):
        # load common keys from prior step
        with open(self.common_fp, 'r') as fid:
            common_keys = json.load(fid)
            common_keys = [int(x) for x in common_keys]
        key_lookup = {k: i for i, k in enumerate(common_keys)}
   

        recon_mfp = np.zeros([len(data), len(common_keys)])
        for i, fingerprint in enumerate(data.values()):
            for key, val in fingerprint.items():
                if int(key) in common_keys:
                    recon_mfp[i, key_lookup[key]] = val

        new_data = {'smiles': data.keys()}
        features = recon_mfp.astype(int).T
        new_data.update({f'fragment_{common_keys[i]}': f for i, f in enumerate(features)})
        new_df = pd.DataFrame(new_data)

        return new_df


class StructureEmbedding(Workflow):

    def read(self, filepath):
        df = pd.read_csv(filepath)
        return df

    def is_valid_file(self, filepath):
        try:
            _ = self.read(filepath)
            return True
        except:
            print('invalid file: ', filepath)
            return False

    def write(self, data, filepath):
        data.to_csv(filepath, index=False)
        return filepath

    def transform(self, data):

        feature_cols = [it for it in data.columns if it.find('feature_') == 0]
        x = data.loc[:, feature_cols]

        if self.standardize:
            x = preprocessing.StandardScaler().fit_transform(x)

        # TODO: save the embedding for future use
        pca = decomposition.PCA(n_components=self.n_components)
        z = pca.fit_transform(x.values)
        for i in range(self.n_components):
            data[f'latent_{i+1}'] = z[:, i]

        data.drop(columns=feature_cols, inplace=True)

        return data


class StructureEmbeddingMany(ManyToOneWorkflow):

    def read(self, filepath):
        df = pd.read_csv(filepath)
        return df

    def is_valid_file(self, filepath):
        try:
            _ = self.read(filepath)
            return True
        except:
            print('invalid file: ', filepath)
            return False

    def write(self, data, filepath):
        df, buffer = data
        df.to_csv(filepath, index=False)

        # don't overwrite an existing model def
        try:
            self.pretrained_path
        except AttributeError:
            model_fp = filepath.replace('.csv', '.lzma')
            if not os.path.isfile(model_fp):
                joblib.dump(buffer, model_fp)
            else:
                raise RuntimeError('No pretrained model was specified, but there is already a model definition at the default location: ', model_fp)

        return filepath

    def transform(self, data):

        try:
            scaler, pca = joblib.load(self.pretrained_path)
        except AttributeError:
            scaler = None
            pca = None

        # assumes same feature columns in every file!
        feature_cols = [it for it in data[0].columns if it.find('fragment_') == 0]
        k = 1
        for d in data[1:]:
            this_feat_cols = [it for it in d.columns if it.find('fragment_') == 0]
            if this_feat_cols != feature_cols:
                raise ValueError(f'Found different fragments in {k}')
            k += 1
        df = pd.concat([it.loc[:, feature_cols] for it in data], ignore_index=True)

        if self.standardize:
            if scaler is None:
                scaler = preprocessing.StandardScaler()
                x = scaler.fit_transform(df.values)
            else:
                x = scaler.transform(df.values)
        else:
            scaler = None
            x = df.values

        # TODO: load the embedding in future
        if pca is None:
            pca = decomposition.PCA(n_components=self.n_components)
            z = pca.fit_transform(x)
        else:
            z = pca.transform(x)

        df = pd.concat(data, ignore_index=True)

        for i in range(self.n_components):
            df[f'latent_{i + 1}'] = z[:, i]

        df.drop(columns=feature_cols, inplace=True)

        buffer = [scaler, pca]

        return (df, buffer)
