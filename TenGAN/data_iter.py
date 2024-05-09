
import pdb
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from mol_metrics import Tokenizer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
rdBase.DisableLog('rdApp.error')


# ============================================================================
# Build a dataset, inherited the methods of Dataset, that returns tensors for Genenerator
class GenDataset(Dataset):

    def __init__(self, data, tokenizer, regression_data=None):
        self.data = data
        self.tokenizer = tokenizer
        self.regression_data = regression_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smile = self.data[idx]
        tensor = self.tokenizer.encode(smile)
        if self.regression_data is None:
            return tensor

        return tensor, self.regression_data.loc[idx].values


# ============================================================================
# Definite the data loader for Generator
class GenDataLoader(LightningDataModule):

    def __init__(self, positive_file, train_size=4800, batch_size=64, regression_file=None):
        super().__init__()
        self.tokenizer = Tokenizer()
        self.train_size = train_size
        self.val_size = 200
        self.batch_size = batch_size
        self.positive_file = positive_file
        self.regression_file = regression_file

    # Randomize the same molecules to different SMILES representations for sufficently training (c1cccc([N+]([O-])=O)c1 -> c1ccccc1[N+](=O)[O-])
    def randomize_smiles_atom_order(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        atom_idxs = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atom_idxs)
        mol = Chem.RenumberAtoms(mol, atom_idxs)
        return Chem.MolToSmiles(mol, canonical=False)

    def custom_collate_and_pad(self, batch):
        # Batch is a list of vectorized smiles
        # pdb.set_trace()
        tensors = [torch.tensor(l[0]) for l in batch]
        # Pad the different lengths of tensors to the maximum length (each column is a sequence)
        tensors = torch.nn.utils.rnn.pad_sequence(
            tensors)  # [maxlength, batch_size]
        return tensors, [torch.tensor(l[1]) for l in batch]

    def setup(self):
        # Load data
        self.data = pd.read_csv(
            self.positive_file, nrows=self.val_size + self.train_size, names=['smiles'], header=None)
        self.regression_data = None
        if self.regression_file is not None:
            self.regression_data = pd.read_csv(
                self.regression_file, nrows=self.val_size + self.train_size).reset_index(drop=True)
        # Atom order randomize SMILES
        self.data['smiles'] = self.data['smiles'].apply(
            self.randomize_smiles_atom_order)
        # Initialize Tokenizer
        self.tokenizer.build_vocab()
        # Create splits for train/val
        idxs = np.array(range(len(self.data['smiles'])))
        np.random.shuffle(idxs)
        val_idxs, train_idxs = idxs[:self.val_size], idxs[self.val_size:self.val_size + self.train_size]
        self.train_data = self.data['smiles'][train_idxs]
        self.train_data.reset_index(drop=True, inplace=True)
        self.val_data = self.data['smiles'][val_idxs]
        self.val_data.reset_index(drop=True, inplace=True)
        if self.regression_data is not None:
            self.train_regression_data = self.regression_data.loc[train_idxs]
            self.train_regression_data.reset_index(drop=True, inplace=True)
            self.val_regression_data = self.regression_data.loc[val_idxs]
            self.val_regression_data.reset_index(drop=True, inplace=True)
        else:
            self.train_regression_data = None
            self.val_regression_data = None

    def train_dataloader(self):
        # pdb.set_trace()
        # print(self.train_regression_data)
        # pdb.set_trace()
        dataset = GenDataset(self.train_data, self.tokenizer,
                             self.train_regression_data)
        # pin_memory=True: speed the dataloading, num_workers: multithreading for dataloading
        return DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, collate_fn=self.custom_collate_and_pad, num_workers=40)

    def val_dataloader(self):
        dataset = GenDataset(self.val_data, self.tokenizer,
                             self.val_regression_data)
        return DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, collate_fn=self.custom_collate_and_pad, shuffle=False, num_workers=40)


# ============================================================================
# Build a dataset, inherited the methods of Dataset, that returns tensors for Discriminator
class DisDataset(Dataset):

    def __init__(self, pairs, tokenizer):
        self.data, self.labels = pairs['smiles'], pairs['labels']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smile = self.data[idx]
        # Remove the start token and end token
        tensor = self.tokenizer.encode(smile)[1:-1]
        label = self.labels[idx]
        return tensor, label


# ============================================================================
# Definite the data loader for the Discriminator
class DisDataLoader(LightningDataModule):

    def __init__(self, positive_file, negative_file, batch_size=2):
        super().__init__()
        self.tokenizer = Tokenizer()
        self.batch_size = batch_size
        self.positive_file = positive_file
        self.negative_file = negative_file

    def custom_collate_and_pad(self, batch):
        # Zip a batch of data
        # pd.set_trace()
        smiles, labels = zip(*batch)
        # Batch is a list of vectorized smiles
        tensors = [torch.LongTensor(smi) for smi in smiles]
        # Pad the different lengths of tensors to the maximum length (each column is a sequence)
        tensors = torch.nn.utils.rnn.pad_sequence(
            tensors).transpose(0, 1)  # [batch_size, maxlength]
        labels = torch.LongTensor(labels)
        return tensors, labels

    def setup(self):
        # Load data
        self.positive_data = pd.read_csv(self.positive_file, names=['smiles'])
        self.negative_data = pd.read_csv(self.negative_file, names=['smiles'])
        # Keep the unqiue order for the NEGATIVE dataset
        self.negative_data = pd.DataFrame([Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(
            s) is not None else s for s in self.negative_data['smiles']], columns=['smiles'])

        self.data = pd.concat(
            [self.positive_data['smiles'], self.negative_data['smiles']])
        self.labels = pd.DataFrame([1 for _ in range(len(self.positive_data))] + [
                                   0 for _ in range(len(self.negative_data))], columns=['labels'])
        self.pairs = list(zip(self.data, self.labels['labels']))
        # Initialize Tokenizer
        self.tokenizer.build_vocab()
        # Create splits for train/val
        np.random.shuffle(self.pairs)
        self.pairs = pd.DataFrame(self.pairs, columns=['smiles', 'labels'])
        self.train_data = self.pairs[:int(len(self.pairs)*0.9)]
        self.train_data.reset_index(drop=True, inplace=True)
        self.val_data = self.pairs[int(len(self.pairs)*0.9):]
        self.val_data.reset_index(drop=True, inplace=True)

    def train_dataloader(self):
        dataset = DisDataset(self.train_data, self.tokenizer)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, collate_fn=self.custom_collate_and_pad, num_workers=0)

    def val_dataloader(self):
        dataset = DisDataset(self.val_data, self.tokenizer)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, collate_fn=self.custom_collate_and_pad, num_workers=0)
