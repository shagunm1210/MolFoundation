# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
from pathlib import Path
import pandas as pd
import sklearn.model_selection
import torch
import wandb
import pdb
from regression_layer import NNModel
from data_utils import CustomDataset
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel, AutoTokenizer


wandb.init()

#tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
#LLModel = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

LLModel = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

LLModel.to("cuda")
for name, layer in LLModel.named_children():
    print(name, layer)


nnmodel = NNModel(config={"input_size": 768, "embedding_size": 512, "hidden_size": 256, "output_size": 1, "n_layers": 2}).to("cuda")

EPOCHS = 1

wandb.watch(nnmodel, log_freq=100)

# import data (local import, change path to your data)
data_pwd = '/nfs/lambda_stor_01/data/avasan/LLM_Chemistry/MolFoundation_test/t5-chem/MolFoundation/Data'
data_path: Path = Path(f"{data_pwd}/gdb9_sample_10k.csv")
data = pd.read_csv(data_path)

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

        with torch.no_grad():
            outputs = LLModel(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)#labels=labels, 
            print(outputs)
            #loss = outputs["loss"]
            #logits = outputs["logits"]
            encoder = outputs["hidden_states"][-1]

        # train regression head on top of encoder output to label
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # average over second dimension of encoder output to get a single vector for each example
        encoder = encoder.mean(dim=1)

        # pass encoder output to regression head
        nn_outputs = nnmodel(encoder)
        # calculate loss from outputs and ground_truth_y_values
        nn_loss = F.mse_loss(nn_outputs.flatten(), y_regression_values)
        total_loss += nn_loss.item()
        nn_loss.backward()
        optimizer.step()
        running_loss += nn_loss.item()
        if num_of_examples % 100 == 0:
            last_loss = running_loss / 100 # loss per X examples
            print('num_of_examples {} loss: {} %_data_trained : {}'.format(num_of_examples + 1, last_loss, num_of_examples / len(X_train) * 100))
            wandb.log({"num_of_examples": num_of_examples, "train_loss": last_loss})
            running_loss = 0.
        num_of_examples += len(batch["input_ids"])
            

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
            outputs = LLModel(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            print(outputs)
            loss = outputs["loss"]
            logits = outputs["logits"]
            encoder = outputs["encoder_last_hidden_state"]

            # inference regression head on top of encoder output to label
            # average over second dimension of encoder output to get a single vector for each example
            encoder = encoder.mean(dim=1)

            # pass encoder output to regression head
            nn_outputs = nnmodel(encoder)
            nn_loss = F.mse_loss(nn_outputs.flatten(), y_regression_values)
            # add to dictionary
            outputs_dict["ground_truth"].extend(y_regression_values.cpu().numpy())
            outputs_dict["predictions"].extend(nn_outputs.flatten().cpu().detach().numpy())
            total_loss += nn_loss.item()
            nn_loss.backward()
            optimizer.step()
            running_tloss += nn_loss.item()
            if num_of_examples % 100 == 0:
                last_tloss = running_tloss / 100 # loss per X examples
                print('  num_of_examples {} test_loss: {}'.format(num_of_examples + 1, last_tloss))
                wandb.log({"num_of_test_examples": num_of_examples, "test_loss": last_tloss})
                running_tloss = 0.
                # Track best performance, and save the model's state
                if last_tloss < best_vloss:
                    best_vloss = last_tloss
                    model_path = 'model_{}_{}'.format(timestamp, num_of_examples)
                    torch.save(nnmodel.state_dict(), model_path)
        num_of_examples += len(batch["input_ids"])
    return outputs_dict

def generate_parity_plot(ground_truth, predictions):
    plt.scatter(ground_truth, predictions)
    # draw line of best fit
    m, b = np.polyfit(ground_truth, predictions, 1)
    plt.plot(ground_truth, m*ground_truth + b)
    # add labels of correlation coefficient
    # correlation coefficient
    r = np.corrcoef(ground_truth, predictions)[0, 1]
    # pearson's r squared
    r2 = sklearn.metrics.r2_score(ground_truth, predictions)
    plt.legend(["Data", "y = {:.2f}x + {:.2f}; r={}; r2={}".format(m, b, r, r2)], loc="upper left")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title("Ground Truth vs Predictions")
    plt.savefig("zpve_parity_plot.png")


# Train for 1 epoch
epoch_number = 0
for epoch in range(EPOCHS):
    print(f"Epoch {epoch}")
    train_one_epoch(epoch)
    outputs_dict = inference_test_set(epoch)
    epoch_number += 1

# Generate Parity Plot
generate_parity_plot(outputs_dict["ground_truth"], outputs_dict["predictions"])
