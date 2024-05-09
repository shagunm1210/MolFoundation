import torch
import numpy as np
from mol_metrics import *
from generator import PositionalEncoding
from pytorch_lightning import LightningModule


# ============================================================================
# Definite the model that deep copied from the generator for rollout sampling
class OwnModel(LightningModule):

    def __init__(
        self,
        n_tokens, # vocabulary size
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.setup_layers()
    
    def setup_layers(self):
        self.embedding = torch.nn.Embedding(self.n_tokens, self.d_model)
        self.positional_encoder = PositionalEncoding(self.d_model, dropout=self.dropout)
        encoder_layer = torch.nn.TransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout, self.activation)
        encoder_norm = torch.nn.LayerNorm(self.d_model)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, self.num_encoder_layers, encoder_norm)
        self.fc_out = torch.nn.Linear(self.d_model, self.n_tokens)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1) # Define lower triangular square matrix with dim=sz
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, features): # features.shape[0] = max_len
        mask = self._generate_square_subsequent_mask(features.shape[0]).to(features.device) # mask: [max_len, max_len]
        embedded = self.embedding(features)
        positional_encoded = self.positional_encoder(embedded)
        encoded = self.encoder(positional_encoded, mask=mask)
        out_2= self.fc_out(encoded) # [max_len, batch_size, vocab_size]
        return out_2


# ============================================================================
# Rollout object
class Rollout(object):
    
    def __init__(self, gen, roll_model, tokenizer, update_rate, device):
        self.ori_model = gen # Shallow copy: if mdoel's parameters change, ori_model will change
        self.own_model = roll_model
        self.tokenizer = tokenizer
        self.update_rate = update_rate
        self.device = device

    def get_reward(self, samples, rollsampler, rollout_num, dis, dis_lambda = 0.5, properties = None):
        """
            - samples: a batch of generated SMILES (Be NOT decoded yet)
            - rollout_num: the number of rollout times
            - dis: discrimanator model 
            - dis_lambda: if 0: Naive RL, elif 1: SeqGAN
        """
        encoded = [torch.tensor(self.tokenizer.encode(s)) for s in samples] # Within the start and end token
        paded = torch.nn.utils.rnn.pad_sequence(encoded).squeeze().to(self.device) # [max_len, batch_size]
        seq_len, batch_size = paded.size()
        dis.to(self.device)
        # Inactivate the dropout layer
        dis.eval()
        rewards = [] # Save rewards of the generated SMILES
        init = 2 # Start from the second letter (after the start token and the first action)

        for i in range(rollout_num):
            already = [] # Delete the traversed SMILES
            # Generate SMILES based on the given sub-SMILES
            for given_num in range(init, seq_len):
                data = paded[0:given_num] # [given_num, batch_size]
                generated_smiles = rollsampler.sample(data) # Len of smiles: batch_size
                generated_encoded = [torch.tensor(self.tokenizer.encode(s))[1:-1] for s in generated_smiles] # Remove the start token and end token and as the input of dis
                generated_paded = torch.nn.utils.rnn.pad_sequence(generated_encoded).squeeze().transpose(0, 1).to(self.device) # [batch_size, max_len]
                gind = np.array(range(generated_paded.size(0))) # batch_size
                pred = dis.forward(generated_paded) # [batch_size, 2]
                pred = torch.nn.functional.softmax(pred, dim=1) # [batch_size, 2]
                # Probability of real class
                pred = pred.data[:, 1].cpu().numpy() 
                if dis_lambda != 1.:
                    pred = dis_lambda * pred
                    # Delete sequences that are already finished, and add their rewards
                    for k, r in reversed(already):
                        del generated_smiles[k]
                        gind = np.delete(gind, k, 0)
                        pred[k] += (1 - dis_lambda) * r
                    # If there are still seqs, calculate rewards
                    if len(generated_smiles): # batch_size
                        pct_unique = len(list(set(generated_smiles))) / float(len(generated_smiles))
                        weights = np.array([pct_unique / float(generated_smiles.count(sm)) for sm in generated_smiles])
                        vals = reward_fn(properties, generated_smiles)
                        rew = vals * weights
                    # Add the just calculated rewards
                    for k, r in zip(gind, rew):
                        pred[k] += (1 - dis_lambda) * r
                    # Choose the seqs finished in the last iteration
                    for j, k in enumerate(gind): # k: real idx of gind
                    	if paded[given_num-1][k] == self.tokenizer.char_to_int[self.tokenizer.end]:
                        	already.append((k, rew[j]))                            
                    already = sorted(already, key = lambda el: el[0]) 
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[given_num - init] += pred
            # For the last token 
            last_encoded = [torch.tensor(self.tokenizer.encode(s))[1:-1] for s in samples]
            last_paded = torch.nn.utils.rnn.pad_sequence(last_encoded).squeeze().to(self.device)
            pred = dis.forward(last_paded.transpose(0, 1)) # [batch_size, max_len] -> [batch_size, 2]
            pred = torch.nn.functional.softmax(pred, dim=1).cpu()
            pred = pred.data[:, 1].numpy()
            if dis_lambda != 1.:
                pred = dis_lambda * pred
                pct_unique = len(list(set(samples))) / float(len(samples))
                weights = np.array([pct_unique / float(samples.count(s)) for s in samples])                
                vals = reward_fn(properties, samples)
                rew = vals * weights
                pred += (1 - dis_lambda) * rew
            if i == 0:
                rewards.append(pred)
            else:
                rewards[-1] += pred
        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num) # [batch_size, seq_len]
        rewards = rewards - np.mean(rewards)
        # Activate the dropout layer
        dis.train()
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]

