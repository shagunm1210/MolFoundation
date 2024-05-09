import math
import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning
from generator import PositionalEncoding
from pytorch_lightning import LightningModule


# ============================================================================
# Definition of the Discriminator model
class DiscriminatorModel(LightningModule):
    
    def __init__(
        self,
        n_tokens, # vocabulary size
        d_model=256,
        nhead=8,
        num_encoder_layers=2,
        dim_feedforward=200,
        dropout=0.1,
        max_lr=1e-3,
        epochs=10,
        pad_token=0,
        dis_wgan=True,
        minibatch=True,
    ):
        super().__init__()
        assert d_model % nhead == 0, "nheads must divide evenly into d_model" 
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_lr = max_lr
        self.epochs = epochs
        self.pad_token = pad_token
        self.dis_wgan = dis_wgan
        self.minibatch = minibatch
        self.setup_layers()

    # Initialize parameters with truncated normal distribution for the classifer 
    def truncated_normal_(self,tensor,mean=0,std=0.1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size+(4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    # Apply WGAN loss to alleviate the mode-collapse problem
    def wgan_loss(self, outputs, labels):
        """
            outputs: the digit outputs of discriminator forward function with size [batch_size, 2]
            labels: labels of the discriminator with size [batch_size]
        """
        assert len(labels.shape) == 1
        assert outputs.shape[0] == labels.shape[0]
        # partation the outputs according to the label 0 and 1
        neg, pos = [outputs[labels == i] for i in range(2)]
        w_loss = torch.abs(torch.sum(neg) / (neg.shape[0] + 1e-10) - torch.sum(pos) / (pos.shape[0] + 1e-10))
        return w_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.max_lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.max_lr, 
            total_steps=None, 
            epochs=self.epochs, 
            steps_per_epoch=len(self.train_dataloader()),
            pct_start=6/self.epochs, 
            anneal_strategy='cos', 
            cycle_momentum=True, 
            base_momentum=0.85, 
            max_momentum=0.95,
            div_factor=1e3, 
            final_div_factor=1e3, 
            last_epoch=-1)        
        scheduler = {"scheduler": scheduler, "interval" : "step" }
        return [optimizer], [scheduler]
    
    def setup_layers(self):
        self.embedding = torch.nn.Embedding(self.n_tokens, self.d_model)
        self.positional_encoder = PositionalEncoding(self.d_model, dropout=self.dropout)
        encoder_layer = torch.nn.TransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, self.num_encoder_layers)
        if self.minibatch:
            self.classifier = torch.nn.Linear(self.d_model+1, 2)
        else:
            self.classifier = torch.nn.Linear(self.d_model, 2)
        self.criterion = torch.nn.CrossEntropyLoss()

        # Intialize the parameters of the discriminator
        for name, param in self.named_parameters():
            if len(param.data.size()) > 1:
                torch.nn.init.xavier_uniform_(param.data)
            if name.startswith('classifier'):
                param.data = self.truncated_normal_(param.data)

    def padding_mask(self, src): # src:[batch_size, maxlength]
        return (src == self.pad_token).transpose(0, 1) # [maxlength, batch_size]

    # Omit the effect of padding token
    def masked_mean(self, encoded, mask):
        """
            encoded: output of TransformerEncoder with size [batch_size, maxlength, d_model]
            mask: output of _padding_mask with size [maxlength, batch_size]: if pad: True, else False
            return: mean of the encoded according to the non-zero/True mask [batch_size, d_model]
        """
        non_mask = mask.transpose(0,1).unsqueeze(-1) == False # [batch_size, maxlength, 1] if Pad: 0, else 1
        masked_encoded = encoded * non_mask # [batch_size, maxlength, d_model]
        ave = masked_encoded.sum(dim=1) / non_mask.sum(dim=1) # [batch_size, d_model]

        return ave

    # Apply mini-batch discrimination to alleviate the mode-collapse problem
    def minibatch_std(self, x):
        """
            x: output of the middle layer of Discriminator with size [batch_size, d_model]
            return: contains the mean of the std information of x
        """
        size = list(x.size())
        size[1] = 1
        # Compute std according to the batch_size direction
        std = torch.std(x, dim=0, unbiased=False) # [d_model+1]
        mean = torch.mean(std) # length of one

        return torch.cat((x, mean.repeat(size)), dim=1) # [batch_size, d_model+1]
    
    def forward(self, features): #[batch_size, maxlength]
        paded_mask = self.padding_mask(features)
        embedded = self.embedding(features) * math.sqrt(self.d_model) #[batch_size, maxlength, d_model]
        positional_encoded = self.positional_encoder(embedded) #[batch_size, maxlength, d_model]
        encoded = self.encoder(positional_encoded) # [batch_size, maxlength, d_model]
        masked_out = self.masked_mean(encoded, paded_mask) # [batch_size, d_model]

        # If true: apply mini-batch discriminator
        if self.minibatch:
            masked_out = self.minibatch_std(masked_out)
        # If true: apply WGAN
        if self.dis_wgan:
            weight_loss = torch.sum(self.classifier.weight**2) / 2.
            bias_loss = torch.sum(self.classifier.bias**2) / 2.
            self.l2_loss = weight_loss + bias_loss
        out = self.classifier(masked_out) #[batch_size, 2]

        torch.set_printoptions(threshold=np.inf)
        np.set_printoptions(threshold=np.inf)
        if torch.sum(torch.isnan(out)) != 0:
            print('paded_mask:', paded_mask)
            print('embedded:', embedded)
            print('positional_encoded:', positional_encoded)
            print('encoded:', encoded)
            print('masked_out:', masked_out)
            print('out:', out)

        return out

    def step(self, batch):
        inputs, labels = batch # inputs:[batch_size, maxlength], labels: [batch_size]
        outputs = self.forward(inputs) #[batch_size, 2]
        if self.dis_wgan:
            # Compute WGAN loss
            w_loss = self.wgan_loss(outputs, labels)
            loss = w_loss + self.l2_loss * 0.2 
        else:
            # Compute cross-entropy loss for GAN
            loss = self.criterion(outputs, labels)
        # Compute accuracy for the classifier 
        pred = outputs.data.max(1)[1] # Indices of max elements
        acc = pred.eq(labels.data).cpu().sum() / len(labels)
        return loss, acc
    
    def training_step(self, batch, batch_idx):
        self.train()
        loss, acc = self.step(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        loss, acc = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss