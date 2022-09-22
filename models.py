import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from transformers.modeling_outputs import BaseModelOutput
from transformers import RobertaConfig
from ood_model import  RobertaForSequenceClassification
#MAX_LEN is the allowed length for each sentence
MAX_LEN = 40
# #VOC_SIZE is the vocabulary size for our tokenizer
VOC_SIZE = 50265
class BartGenerationNet(pl.LightningModule):

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
    
class NaiveBertNet(pl.LightningModule):

    def __init__(self, bert, fc1 = 200):
        super(NaiveBertNet, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(768,2)
        self.drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(fc1,2)
        self.sf = nn.Softmax(dim = 1)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.bert(x)[0][:,0,:]
        x = self.fc1(x)
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
        target =  target.flatten()
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer
    
class NaiveRobertaNet(pl.LightningModule):

    def __init__(self, bert, fc1 = 200):
        super(NaiveRobertaNet, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(768,2)
        self.drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(fc1,2)
        self.sf = nn.Softmax(dim = 1)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.bert(x, output_hidden_states = True).hidden_states[-1][:,0,:]
        x = self.fc1(x)
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
        target =  target.flatten()
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer  

class BertEPNet(pl.LightningModule):

    def __init__(self, bert, biased_model):
        super(BertEPNet, self).__init__()
        self.bert = bert
        self.biased_model = biased_model
        self.fc1 = nn.Linear(768,2)
        self.drop = nn.Dropout(p=0.4)
        self.sf = nn.Softmax(dim = -1)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.bert(x)[0][:,0,:]
        x = self.fc1(x)
        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        inputs, sentence_o, target = train_batch
        target =  target.flatten()
        biased_dist = self.sf(self.biased_model(sentence_o))
        predicted_dist = self.sf(self.forward(inputs))
        output_dist = torch.log(biased_dist)+torch.log(predicted_dist)
        loss = self.criterion(output_dist, target)
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
        inputs, sentence_o, target = val_batch
        target =  target.flatten()
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.bert.parameters())+ list(self.bert.parameters()), lr=2e-5)
        return optimizer
    


class NaiveBartNet(pl.LightningModule):

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
        target =  target.flatten()
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer
class OODRobertaNet(pl.LightningModule):

    def __init__(self, config = None):
        super(OODRobertaNet, self).__init__()
        if config == None:
            config = RobertaConfig.from_pretrained('roberta-base', num_labels=2)
        self.model = RobertaForSequenceClassification(config)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,\
        input_ids=None,\
        attention_mask=None,\
        labels = None):
        outputs = self.model.roberta(
            input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        logits, pooled = self.model.classifier(sequence_output)
        return logits


    def training_step(self, train_batch, batch_idx):
        inputs, target = train_batch
        batch = train_batch
        outputs = self.forward(batch['input_ids'])
        loss = self.criterion(outputs, batch['labels'].flatten())
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):

        batch = val_batch
        outputs = self.forward(batch['input_ids'])
        loss = self.criterion(outputs, batch['labels'].flatten())
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer  
    
class MultiBertNet(pl.LightningModule):

    def __init__(self, bert, fc1 = 200):
        super(MultiBertNet, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(768,2)
        self.fc2 = nn.Linear(768,3)
        self.sf = nn.Softmax(dim = 1)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.bert(x)[0][:,0,:]
        pred1 = self.fc1(x)
        pred2 = self.fc2(x)
        return pred1,pred2

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        inputs, target = train_batch
        target =  target.flatten()
        para_indexes = torch.where(target < 3)
        nli_indexes = torch.where(target >= 3)
        para_target = target[para_indexes]
        nli_target = target[nli_indexes] - 3
        pred1, pred2 = self.forward(inputs)
        para_pred = pred1[para_indexes]
        nli_pred = pred2[nli_indexes]
        loss = self.criterion(para_pred, para_target) + self.criterion(nli_pred, nli_target)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, target = val_batch
        target =  target.flatten()
        para_indexes = torch.where(target < 3)
        nli_indexes = torch.where(target >= 3)
        para_target = target[para_indexes]
        nli_target = target[nli_indexes] - 3
        pred1, pred2 = self.forward(inputs)
        para_pred = pred1[para_indexes]
        nli_pred = pred2[nli_indexes]
        loss = self.criterion(para_pred, para_target) + self.criterion(nli_pred, nli_target)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer