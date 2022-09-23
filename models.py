import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
import torch.nn as nn
from utils import MAX_LEN, VOC_SIZE
class BartGenerationNet(pl.LightningModule):
    """
    the class where we use Bart for conditional generation
    """

    def __init__(self, bart, fc1 = 2):
        super(BartGenerationNet, self).__init__()
        self.bart = bart
        self.drop = nn.Dropout(p=0.5)
        # self.fc2 = nn.Linear(64,18)
        self.criterion = nn.CrossEntropyLoss()
        self.sf = nn.Softmax(dim = 1)
        
    def forward(self, x, decoder_input_ids):
        decoder_input_ids = self.shift_tokens_right(decoder_input_ids, 1)
        x = self.bart(x, decoder_input_ids = decoder_input_ids).logits
        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        inputs, target = train_batch
        logits= self.forward(inputs,target).reshape(-1, VOC_SIZE)
        target =  target.flatten()
        loss = self.criterion(logits, target)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss
    def shift_tokens_right(self, input_ids, pad_token_id):
        """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
          This is taken directly from modeling_bart.py
        """
        prev_output_tokens = input_ids.clone()
        index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = input_ids[:, :-1]
        return prev_output_tokens

    def validation_step(self, val_batch, batch_idx):
        inputs, target = val_batch
        logits= self.forward(inputs,target).reshape(-1, VOC_SIZE)
        target =  target.flatten()
        loss = self.criterion(logits, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer

class NaiveBartNet(pl.LightningModule):
    """
    the model class where we use bart for sequence classification
    """

    def __init__(self, bart, fc1 = 200):
        super(NaiveBartNet, self).__init__()
        self.bart = bart
        self.fc1 = nn.Linear(768,2)
        self.drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(fc1,2)
        self.sf = nn.Softmax(dim = 1)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x_input = x
        indexes = torch.sum(x == 1, dim = 1)
        # tokens2 = shift_tokens_right(tokens2, 1)
        bart_output = self.bart(input_ids = x,decoder_input_ids = x, output_hidden_states=True)
        x = bart_output.decoder_hidden_states[-1]
        #choose the </s> token
        x = torch.cat([x[i, 2*MAX_LEN -1 - index,:].reshape(1,-1) for i, index in enumerate(indexes)], dim = 0)
        # print(torch.cat([x_input[i, 2*MAX_LEN -1 - index].reshape(1,-1) for i, index in enumerate(indexes)], dim = 0))
        # x = torch.max(x, dim = 1)[0]

        # x = torch.mean(x, dim = 1)
        x = self.fc1(x)

        # logits = bart_output.logits
        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        inputs, target = train_batch
        target =  target.flatten()
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, target)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, target = val_batch
        target =  target.flatten()
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer