# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import pandas as pd
import sklearn

# tokenizer = AutoTokenizer.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")
# model = AutoModelForSeq2SeqLM.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")

# import data (local import, change path to your data)
data_path: Path = Path("../../../Datasets/qm9/gdb9_wSmiles.csv")
data = pd.read_csv(data_path)

# split data
X_train, X_test = sklearn.model_selection.train_test_split(data["SMILES"], test_size=0.2, random_state=42)
Y_train, Y_test = sklearn.model_selection.train_test_split(data["zpve"], test_size=0.2, random_state=42)

# tokenize data
# testing