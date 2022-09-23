import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoModel
from transformers import BartForConditionalGeneration
from models import BartGenerationNet, NaiveBartNet, MAX_LEN
from utils import make_generative_data, make_discriminative_data

def get_generative_model(option, train_pairs, test_pairs, max_epochs = 3, MAX_LEN = MAX_LEN):
    """
    Train a generation model that generates question2 conditioned question1 from train_pairs
    This function is used for positive model and negative model
    """
    train_sentence_o, train_sentence_p = make_generative_data(train_pairs, option = option)
    dev_sentence_o, dev_sentence_p = make_generative_data(test_pairs, option = option)
    train_dataset = torch.utils.data.TensorDataset(train_sentence_o, train_sentence_p)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last = True)
    dev_dataset = torch.utils.data.TensorDataset(dev_sentence_o,  dev_sentence_p)
    dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=True, drop_last = True)
    bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    model = BartGenerationNet(bart)
    trainer = pl.Trainer(devices = 1, accelerator = 'gpu',
                    auto_scale_batch_size= "power" , max_epochs = max_epochs, benchmark = True,auto_lr_find=True)
    trainer.fit(model, train_dataloader, dev_dataloader)
    return model

def get_distribution_model(train_pairs, test_pairs, empty_token, max_epochs = 3, MAX_LEN = MAX_LEN):
    """
    Train a generation model that generates (question1 concat question2) from train_pairs without condition
    This function is used for the distribution model
    """
    train_sentence_p = torch.cat([pair.tokens_combined.reshape(1,MAX_LEN*2) for pair in train_pairs], dim = 0)
    train_sentence_o = torch.cat([empty_token.reshape(1,MAX_LEN*2) for pair in train_pairs], dim = 0)
    dev_sentence_p = torch.cat([pair.tokens_combined.reshape(1,MAX_LEN*2) for pair in test_pairs], dim = 0)
    dev_sentence_o = torch.cat([empty_token.reshape(1,MAX_LEN*2) for pair in test_pairs], dim = 0)
    train_dataset = torch.utils.data.TensorDataset(train_sentence_o, train_sentence_p)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last = True)
    dev_dataset = torch.utils.data.TensorDataset(dev_sentence_o,  dev_sentence_p)
    dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=True, drop_last = True)
    bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    model = BartGenerationNet(bart)
    trainer = pl.Trainer(devices = 1, accelerator = 'gpu',
                    auto_scale_batch_size= "power" , max_epochs = max_epochs, benchmark = True,auto_lr_find=True)
    trainer.fit(model, train_dataloader, dev_dataloader)
    return model

def get_discriminative_model(train_pairs, test_pairs, max_epochs = 3):
    """
    Train a discriminative_model that predicts the label of (question1 concat question2) from train_pairs
    This function is used for the discriminative model
    """
    features, labels = make_discriminative_data(train_pairs)
    dev_features, dev_labels = make_discriminative_data(test_pairs)
    train_dataset = torch.utils.data.TensorDataset(features, labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last = True)
    dev_dataset = torch.utils.data.TensorDataset(dev_features,  dev_labels)
    dev_dataloader = DataLoader(dev_dataset, batch_size=32, drop_last = True)
    bart = AutoModel.from_pretrained('facebook/bart-base')
    model = NaiveBartNet(bart)
    trainer = pl.Trainer(devices = 1, accelerator = 'gpu',
                    auto_scale_batch_size= "power" , max_epochs = max_epochs, benchmark = True,auto_lr_find=True)
    trainer.fit(model, train_dataloader, dev_dataloader)
    return model
