# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pathlib import Path
import pandas as pd
import sklearn.model_selection
import torch
# import wandb
import pdb
from models import NNModel, chemberta_for_regression
from data_utils import CustomDataset
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# wandb.init()

tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
LLModel = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
LLModel.to("cuda")

nnmodel = NNModel(config={"input_size": 768, "embedding_size": 512, "hidden_size": 256, "output_size": 1, "n_layers": 2}).to("cuda")

# finetune model
FTModel = chemberta_for_regression(LLModel, nnmodel)
EPOCHS = 1

# wandb.watch(nnmodel, log_freq=100)

# import data (local import, change path to your data)
#data_path: Path = Path("../../../Datasets/qm9/gdb9_sample_10k.csv")
#data = pd.read_csv(data_path)
data = pd.read_csv('gdb9_sample_10k_zpve.csv')

# results path
results_path = Path("./results")

# split data
X_train, X_test = sklearn.model_selection.train_test_split(data["SMILES"], test_size=0.2, random_state=42)
Y_train, Y_test = sklearn.model_selection.train_test_split(data["zpve"], test_size=0.2, random_state=42)

# convert pandas dataframe to list for tokenization
X_train = X_train.tolist()
X_test = X_test.tolist()

# convert pandas dataframe to tensor
Y_train = torch.tensor(Y_train.tolist())
Y_test = torch.tensor(Y_test.tolist())

# create CustomDataset object
training_set = CustomDataset(tokenizer, X_train, Y_train, max_input_length=512, max_target_length=512)
test_set = CustomDataset(tokenizer, X_test, Y_test, max_input_length=512, max_target_length=512)

# create dataloader
train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)

# Initialize optimizer
optimizer = torch.optim.Adam(nnmodel.parameters(), lr=1e-5)

# Timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Training Loop
# Output of the model is a dictionary with keys "loss" and "logits"
def train_one_epoch(epoch_index):
    LLModel.eval()
    running_loss = 0.0
    total_loss = 0
    num_of_examples: int = 0
    for batch in train_dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        y_regression_values = batch["y_regression_values"]

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        ft_output = FTModel(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # calculate loss from outputs and ground_truth_y_values
        ft_loss = F.mse_loss(ft_output.flatten(), y_regression_values)
        total_loss += ft_loss.item()
        ft_loss.backward()
        optimizer.step()
        running_loss += ft_loss.item()
        if num_of_examples % 100 == 0:
            last_loss = running_loss / 100 # loss per X examples
            print('num_of_examples {} loss: {} %_data_trained : {}'.format(num_of_examples, last_loss, num_of_examples / len(X_train) * 100))
            # wandb.log({"num_of_examples": num_of_examples, "train_loss": last_loss})
            running_loss = 0.
        num_of_examples += len(batch["input_ids"])
        # break
            

def inference_test_set(epoch_index):
    LLModel.eval()
    running_tloss = 0.0
    total_tloss = 0
    num_of_examples: int = 0
    # dictionary of all ground_truth and predictions
    outputs_dict = {"ground_truth": [], "predictions": []}
    for batch in test_dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        y_regression_values = batch["y_regression_values"]

        with torch.no_grad():
            ft_output = FTModel(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            ft_loss = F.mse_loss(ft_output.flatten(), y_regression_values)
            # add to dictionary
            outputs_dict["ground_truth"].extend(y_regression_values.cpu().numpy())
            outputs_dict["predictions"].extend(ft_output.flatten().cpu().detach().numpy())
            total_tloss += ft_loss.item()
            running_tloss += ft_loss.item()
            # if num_of_examples % 100 == 0:
            #     last_tloss = running_tloss / 100 # loss per X examples
            #     print('  num_of_examples {} test_loss: {}'.format(num_of_examples + 1, last_tloss))
            # wandb.log({"num_of_test_examples": num_of_examples, "test_loss": total_tloss})
                # running_tloss = 0.
                # # Track best performance, and save the model's state
                # if last_tloss < best_vloss:
                #     best_vloss = last_tloss
                #     model_path = 'model_{}_{}'.format(timestamp, num_of_examples)
                #     torch.save(nnmodel.state_dict(), model_path)
        num_of_examples += len(batch["input_ids"])
        # break

    return outputs_dict

def generate_parity_plot(ground_truth, predictions):
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    plt.scatter(ground_truth, predictions, s=4)
    # draw line of best fit
    m, b = np.polyfit(ground_truth, predictions, 1)
    plt.plot(ground_truth, m*ground_truth + b, color="orange")
    # add labels of correlation coefficient
    # correlation coefficient
    r = np.corrcoef(ground_truth, predictions)[0, 1]
    # pearson's r squared
    r2 = sklearn.metrics.r2_score(ground_truth, predictions)
    plt.legend(["Data", "y = {:.2f}x + {:.2f}; r={}; r2={}".format(m, b, r, r2)], loc="upper left")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title("Ground Truth vs Predictions")
    plt.savefig(results_path / "chemberta-fine-tune-zpve-10K_parity_plot.png")


# Train for 1 epoch
epoch_number = 0
for epoch in range(EPOCHS):
    print(f"Epoch {epoch}")
    train_one_epoch(epoch)
    outputs_dict = inference_test_set(epoch)
    epoch_number += 1

# Generate Parity Plot
generate_parity_plot(outputs_dict["ground_truth"], outputs_dict["predictions"])
