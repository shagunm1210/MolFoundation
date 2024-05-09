
import pdb
import math
import torch
from tqdm import tqdm
import pytorch_lightning
from pytorch_lightning import LightningModule


# ============================================================================
# PositionalEncoding needs to be definited manually, even if the transformer model is called from torch.nn
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
            d_model: the embedded dimension
            max_len: the maximum length of sequences
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # PosEncoder(pos, 2i) = sin(pos/10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # PosEncoder(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            x: the sequence fed to the positional encoder model with the shape [sequence length, batch size, embedded dim].
        """
        x = x + \
            self.pe[:x.size(
                0), :]  # [max_len, batch_size, d_model] + [max_len, 1, d_model]
        return self.dropout(x)


# ============================================================================
# Definition of the Generator model
class GeneratorModel(LightningModule):

    def __init__(
        self,
        n_tokens,  # vocabulary size
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        max_length=1000,
        max_lr=1e-3,
        epochs=50,
        num_tasks=0
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.max_length = max_length
        self.max_lr = max_lr
        self.epochs = epochs
        self.num_tasks = num_tasks
        self.setup_layers()
        self.double()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=None,
            epochs=self.epochs,
            steps_per_epoch=len(self.train_dataloader())*2,
            pct_start=6/self.epochs,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=1e3,
            final_div_factor=1e3,
            last_epoch=-1)

        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]

    def setup_layers(self):
        self.embedding = torch.nn.Embedding(self.n_tokens, self.d_model)
        self.positional_encoder = PositionalEncoding(
            self.d_model, dropout=self.dropout)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            self.d_model, self.nhead, self.dim_feedforward, self.dropout, self.activation)
        encoder_norm = torch.nn.LayerNorm(self.d_model)
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, self.num_encoder_layers, encoder_norm)
        self.fc_out = torch.nn.Linear(self.d_model, self.n_tokens)
        self.fine_tuning_module = torch.nn.ModuleList()
        for i in range(self.num_tasks):
            regression_block = [
                torch.nn.Linear(self.d_model, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1)
            ]
            regression_layers = torch.nn.Sequential(*regression_block)
            self.fine_tuning_module.append(regression_layers)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(
            0, 1)  # Define lower triangular square matrix with dim=sz
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, features):
        mask = self._generate_square_subsequent_mask(
            features.shape[0]).to(self.device)  # mask: [max_len, max_len]
        embedded = self.embedding(features)
        positional_encoded = self.positional_encoder(embedded)
        encoded = self.encoder(positional_encoded, mask=mask)
        return encoded

    def forward(self, encoded):  # features.shape[0] = max_len
        out_2 = self.fc_out(encoded)  # [max_len, batch_size, vocab_size]
        return out_2

    def forward_fine_tune(self, encoded, idx):
        y_pred = self.fine_tuning_module[idx](encoded)
        return y_pred

    def step(self, batch_smiles, batch_regression):
        batch_smiles = batch_smiles.to(self.device)
        encoded = self.encode(batch_smiles[:-1])
        avg_encoded = torch.mean(encoded, dim=0)
        prediction = self.forward(encoded=encoded)  # Skipping the last char
        fine_tune_loss = 0
        for idx in range(len(batch_regression[0])):
            y_pred = self.forward_fine_tune(avg_encoded, idx)
            y_true = torch.stack([x[idx] for x in batch_regression])
            # pdb.set_trace()
            fine_tune_loss += torch.nn.functional.mse_loss(y_pred, y_true)
        loss = torch.nn.functional.cross_entropy(prediction.transpose(0, 1).transpose(
            1, 2), batch_smiles[1:].transpose(0, 1))  # Skipping the first char
        return loss * fine_tune_loss

    def training_step(self, batch, batch_idx):
        self.train()
        # pdb.set_trace()
        smiles_batch = batch[0]
        regression_batch = batch[1]
        loss = self.step(smiles_batch, regression_batch)
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        # pdb.set_trace()
        smiles_batch = batch[0]
        regression_batch = batch[1]
        loss = self.step(smiles_batch, regression_batch)
        self.log('val_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss


# ============================================================================
# Sampling "n" likely-SMILES from the GeneratorModel
class GenSampler():
    def __init__(self, model, tokenizer, batch_size, max_len):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len

    # Sampling a batch of samples by the trained generator
    def sample(self, data=None):
        self.model.eval()
        # Judge whether each sequence in a batch is finished according to the tokenizer.end
        finished = [False] * self.batch_size
        sample_tensor = torch.zeros(
            (self.max_len, self.batch_size), dtype=torch.long).to(self.model.device)
        # The first token is the start char
        sample_tensor[0] = self.tokenizer.char_to_int[self.tokenizer.start]
        with torch.no_grad():
            if data is None:  # Generate SMILES according to the pre-trained Generator
                init = 1
            else:  # Generate sub-SMILES according to the rollout function
                sample_tensor[:len(data)] = data  # [max_len, batch_size]
                init = len(data)

            for i in range(init, self.max_len):
                # Assign the initial sub-SMILES to tensor
                tensor = sample_tensor[:i]
                # The final token as the result
                logits = self.model.forward(tensor)[-1]
                probabilities = torch.nn.functional.softmax(
                    logits, dim=1).squeeze()
                sampled_char = torch.multinomial(
                    probabilities, 1)  # [batch_size, 1]

                for idx in range(self.batch_size):
                    if finished[idx]:
                        sampled_char[idx,
                                     0] = self.tokenizer.char_to_int[self.tokenizer.end]
                    if sampled_char[idx, 0] == self.tokenizer.char_to_int[self.tokenizer.end]:
                        finished[idx] = True

                sample_tensor[i] = sampled_char.squeeze()
                if all(finished):
                    break

        smiles = ["".join(self.tokenizer.decode(sample_tensor[:, i].squeeze(
        ).detach().cpu().numpy())).strip("^$ ") for i in range(self.batch_size)]
        self.model.train()
        return smiles

    # Sampling "n" samples by the trained generator
    def sample_multi(self, n, filename=None):
        samples = []
        for _ in tqdm(range(int(n / self.batch_size))):
            batch_sample = self.sample()
            samples.extend(batch_sample)
        # Write the "n" samples into file
        if filename:
            with open(filename, 'w') as fout:
                for s in samples:
                    fout.write('{}\n'.format(s))
        return samples
