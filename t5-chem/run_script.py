# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import pandas as pd
import sklearn.model_selection
import torch

tokenizer = AutoTokenizer.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")
# model = AutoModelForSeq2SeqLM.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")

# import data (local import, change path to your data)
data_path: Path = Path("../../../Datasets/qm9/gdb9_wSmiles.csv")
data = pd.read_csv(data_path)

# split data
X_train, X_test = sklearn.model_selection.train_test_split(data["SMILES"], test_size=0.2, random_state=42)
Y_train, Y_test = sklearn.model_selection.train_test_split(data["zpve"], test_size=0.2, random_state=42)

# convert pandas dataframe to list for tokenization
X_train = X_train.tolist()
X_test = X_test.tolist()

# convert pandas datafram to tensor
Y_train = torch.tensor(Y_train.tolist())
Y_test = torch.tensor(Y_test.tolist())

# tokenize data
X_train = tokenizer(X_train, return_tensors="pt", padding=True, truncation=True)
X_test = tokenizer(X_test, return_tensors="pt", padding=True, truncation=True)
print(X_train, X_test)

# create dataloader
train_dataset = torch.utils.data.TensorDataset(X_train, Y_train, batch_size=8, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(X_test, Y_test, batch_size=8, shuffle=False)
