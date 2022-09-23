import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import metrics
from scipy.stats import exponweib
import pytorch_lightning as pl
from transformers import AutoModel
from transformers import BartForConditionalGeneration
from models import BartGenerationNet, NaiveBartNet, MAX_LEN, VOC_SIZE
class Question_pair():
  """
  the class used to encode the total information of a quora question pair
  """
  def __init__(self, id, question1, question2, is_duplicate):
    self.id = id
    self.question1 = question1
    self.question2 = question2
    self.is_duplicate = is_duplicate
    self.tokens1 = None
    self.tokens2 = None
    self.tokens_combined = None
  
  def tokenize_self(self, tokenizer, MAX_LEN):
    encoded_input1 = tokenizer(self.question1, max_length = MAX_LEN,
                              truncation = True, padding = "max_length",
                              add_special_tokens = True, return_token_type_ids = False,\
                              return_attention_mask = False, return_tensors = 'pt')['input_ids'].flatten()
    encoded_input2 = tokenizer(self.question2, max_length = MAX_LEN,
                              truncation = True, padding = "max_length",
                              add_special_tokens = True, return_token_type_ids = False,\
                              return_attention_mask = False, return_tensors = 'pt')['input_ids'].flatten()
    encoded_inputs = tokenizer(self.question1, self.question2, max_length = MAX_LEN*2,
                              truncation = True, padding = "max_length",
                              add_special_tokens = True, return_token_type_ids = False,\
                              return_attention_mask = False, return_tensors = 'pt')['input_ids'].flatten()
                          
    self.tokens1 = encoded_input1
    self.tokens2 = encoded_input2
    self.tokens_combined = encoded_inputs
def get_pairs(file_path, index_id, index_question1, index_question2, index_duplicate,\
              tokenizer, MAX_LEN = MAX_LEN, limit = 50000):
  """
  use this method to load qqp, pit, and paws dataset
  """
  #prepare training examples for pit
  df = pd.read_csv(file_path, sep = '\t')
  id_column = df.iloc[:limit,index_id]
  question1_column = df.iloc[:limit,index_question1]
  question2_column = df.iloc[:limit,index_question2]
  is_duplicate_column = df.iloc[:limit,index_duplicate]
  pairs = []
  for i, id in enumerate(id_column):
    #this try is for pit dataset
    try:
      if int(is_duplicate_column[i][1]) >= 4:
        is_duplicate = 1
      elif int(is_duplicate_column[i][1]) <= 1:
        is_duplicate = 0
      else:
        continue
    except IndexError:
      is_duplicate = is_duplicate_column[i]
    pairs.append(Question_pair(id, question1_column[i], question2_column[i],\
                                 is_duplicate))
  #filter abnormal examples
  pairs = list(filter(lambda x: isinstance(x.question1, str) and isinstance(x.question2, str), pairs))
  for i,pair in enumerate(pairs):
    pair.tokenize_self(tokenizer, MAX_LEN)
  return pairs


def get_pit_test_pairs(tokenizer, MAX_LEN = MAX_LEN):
  """
  use this method to get the test data for pit
  """
  df_data = pd.read_csv('pit/test.data', sep = '\t')
  df_label = pd.read_csv('pit/test.label', sep = '\t')
  id_column = df_data.iloc[:,0]
  question1_column = df_data.iloc[:,2]
  question2_column = df_data.iloc[:,3]
  is_duplicate_column = df_label.iloc[:,1]
  pairs = []
  for i, id in enumerate(id_column):
    #this try is for pit dataset
    try:
      if float(is_duplicate_column[i]) >= 0.8 :
        is_duplicate = 1
      elif float(is_duplicate_column[i]) <= 0.2 :
        is_duplicate = 0
      else:
        continue
    except IndexError:
      is_duplicate = is_duplicate_column[i]
    pairs.append(Question_pair(id, question1_column[i], question2_column[i],\
                                 is_duplicate))
  #filter abnormal examples
  pairs = list(filter(lambda x: isinstance(x.question1, str) and isinstance(x.question2, str), pairs))
  for i,pair in enumerate(pairs):
    pair.tokenize_self(tokenizer, MAX_LEN)
  return pairs
def make_generation_data(pairs, option = 'positive', MAX_LEN = MAX_LEN):
  """
  make generation data
  accepts parameters: positive, negative, all
  """
  sentence_p = []
  sentence_o = []
  for pair in pairs:
    if (pair.is_duplicate == 1 and option == 'positive') or\
    (pair.is_duplicate == 0 and option == 'negative') or\
    (option == 'all'):
      sentence_p.append(pair.tokens2.reshape(1,MAX_LEN))
      sentence_o.append(pair.tokens1.reshape(1,MAX_LEN))
  return torch.cat(sentence_o, dim = 0), torch.cat(sentence_p, dim = 0)

def generate_multitask_data(pairs, MAX_LEN = MAX_LEN):
  """
  generate data for multitask training
  """
  features = []
  sentence_p = []
  sentence_o = []
  labels = []
  for pair in pairs:
    sentence_o.append(pair.tokens1.reshape(1, MAX_LEN))
    features.append(pair.tokens_combined.reshape(1,MAX_LEN*2))
    labels.append(pair.is_duplicate)
  return torch.cat(features, dim = 0), torch.Tensor(labels).long().reshape(-1,1),torch.cat(sentence_o, dim = 0)

def pairs_auc(test_pairs, all_scores):
  """
  calculate the auc scores
  """
  pair2scores = {}
  for i,pair in enumerate(test_pairs):
    pair2scores[pair] =  all_scores[i]
  sorted_pairs = sorted(test_pairs, key = lambda x:pair2scores[x])
  pred = sorted(all_scores)
  y = [2 - pair.is_duplicate  for pair in sorted_pairs]
  fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
  return metrics.auc(fpr, tpr)

def model_scorer(cand, ref, model, tokenizer, MAX_LEN = MAX_LEN):
  model.eval()
  model = model.to('cuda')
  criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction = 'mean')
  all_scores = []
  tokens1 = tokenizer(ref, max_length = MAX_LEN,
                              truncation = True, padding = "max_length",
                              add_special_tokens = True, return_token_type_ids = False,\
                              return_attention_mask = False, return_tensors = 'pt')['input_ids']
  tokens2 = tokenizer(cand, max_length = MAX_LEN,
                              truncation = True, padding = "max_length",
                              add_special_tokens = True, return_token_type_ids = False,\
                              return_attention_mask = False, return_tensors = 'pt')['input_ids']
  tokens1 = tokens1.to('cuda')
  tokens2 = tokens2.to('cuda')
  num_batch = tokens1.size(0)//32 
  if tokens1.size(0) % 32 != 0:
        num_batch += 1
  for j in range(num_batch):
    x1 = tokens1[j*32:j*32+32,:]
    x2 = tokens2[j*32:j*32+32,:]
    with torch.no_grad():
        batch_size = x1.size(0)
#         losses = model(x1, x2).cpu().detach().numpy()
        conditional_logits_1 = model(x1, x2).reshape(batch_size,-1, VOC_SIZE)
        conditional_logits_2 = model(x2, x1).reshape(batch_size,-1, VOC_SIZE)
        losses = []
        for i in range(batch_size):
          losses_1 = criterion(conditional_logits_1[i], x2[i])
          losses_2 = criterion(conditional_logits_2[i], x1[i])
          losses.append((losses_1 + losses_2)/2)
          
    all_scores.append(torch.Tensor(losses).reshape(batch_size,-1))
  all_scores = torch.cat(all_scores, dim = 0)
  return all_scores

def bert_scorer(cand, ref, model, tokenizer, MAX_LEN = MAX_LEN):
  """
  use the model (basic bert or roberta model) to assign scores to each cand-ref pair
  Args:
  - :param: `cand` ( list of str): candidate sentences (sentence 1)
  - :param: `ref` ( list of str): reference sentences (sentence 2), of the same length
    as cand
  """
  model.eval()
  model = model.to('cuda')
  criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction = 'mean')
  all_scores = []
  sf = nn.Softmax(dim = -1)
  tokens = tokenizer(cand, ref, max_length = MAX_LEN*2,
                              truncation = True, padding = "max_length",
                              add_special_tokens = True, return_token_type_ids = False,\
                              return_attention_mask = False, return_tensors = 'pt')['input_ids']

  tokens = tokens.to('cuda')
  num_batch = tokens.size(0)//32 
  if tokens.size(0) % 32 != 0:
        num_batch += 1
  for j in range(num_batch):
    x = tokens[j*32:j*32+32,:]
    with torch.no_grad():
        batch_size = x.size(0)
        outputs = sf(model(x).reshape(batch_size, 2))[:,1]
    all_scores.append(outputs.reshape(batch_size,-1))
  all_scores = torch.cat(all_scores, dim = 0)
  return all_scores.cpu()

def distribution_scorer(cand, ref, model, tokenizer):
  model.eval()
  model = model.to('cuda')
  criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction = 'mean')
  all_scores = []
  empty_ref = ['' for c in cand]
  tokens1 = tokenizer(empty_ref, max_length = MAX_LEN*2,
                              truncation = True, padding = "max_length",
                              add_special_tokens = True, return_token_type_ids = False,\
                              return_attention_mask = False, return_tensors = 'pt')['input_ids']
  tokens2 = tokenizer(cand, ref, max_length = MAX_LEN*2,
                              truncation = True, padding = "max_length",
                              add_special_tokens = True, return_token_type_ids = False,\
                              return_attention_mask = False, return_tensors = 'pt')['input_ids']
  tokens1 = tokens1.to('cuda')
  tokens2 = tokens2.to('cuda')
  num_batch = tokens1.size(0)//32 
  if tokens1.size(0) % 32 != 0:
        num_batch += 1
  for j in range(num_batch):
    x1 = tokens1[j*32:j*32+32,:]
    x2 = tokens2[j*32:j*32+32,:]
    with torch.no_grad():
        batch_size = x1.size(0)
        conditional_logits_1 = model(x1, x2).reshape(batch_size,-1, VOC_SIZE)
        losses = []
        for i in range(batch_size):
          losses_1 = criterion(conditional_logits_1[i], x2[i])
          losses.append(losses_1 )
          
    all_scores.append(torch.Tensor(losses).reshape(batch_size,-1))
  all_scores = torch.cat(all_scores, dim = 0)
  return all_scores

def get_embeddings(x, bert):
    """
    use bert to get sentence embeddings
    """
    new_embeddings = []
    bert = bert.to('cuda').eval()
    x = x.to('cuda')
    with torch.no_grad():
        for i in range(0, x.size(0), 32):
            embeddings = bert(x[i:i+32])[0][:,0,:]
            new_embeddings.append(embeddings)
    new_embeddings = torch.cat(new_embeddings, dim = 0)
    return new_embeddings

def get_generative_model(option, train_pairs, test_pairs, max_epochs = 3, MAX_LEN = MAX_LEN):
    train_sentence_o, train_sentence_p = make_generation_data(train_pairs, option = option)
    dev_sentence_o, dev_sentence_p = make_generation_data(test_pairs, option = option)
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
    features, labels, sentence_o = generate_multitask_data(train_pairs)
    dev_features, dev_labels, dev_sentence_o = generate_multitask_data(test_pairs)
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

def robust_predictions(positive_model, negative_model, distribution_model,\
                       dev_pairs, test_pairs, tokenizer, discriminative_model= None, C = 3):
    thresh = .1
    #first fit a weibull distribution to dev set
    candidates1 = [pair.question1 for pair in dev_pairs]
    candidates2 = [pair.question2 for pair in dev_pairs]
    benchmark_scores = distribution_scorer(candidates1, candidates2, distribution_model, tokenizer)
    a,c, loc, scale = exponweib.fit(benchmark_scores)
    
    candidates1 = [pair.question1 for pair in test_pairs]
    candidates2 = [pair.question2 for pair in test_pairs]
    #calculate the lambda for each testing sample (called distribution_weights)
    distribution_scores = distribution_scorer(candidates1, candidates2, distribution_model, tokenizer)
    distribution_weights = 1 - torch.Tensor(exponweib.cdf(distribution_scores,a,c, loc, scale))
    #thresholding
    distribution_weights = torch.where(distribution_weights>thresh, 1, distribution_weights)
    
    #calculate the scores from the positive and negative model
    positive_scores = model_scorer(candidates1, candidates2, positive_model, tokenizer)
    negative_scores = model_scorer(candidates1, candidates2, negative_model, tokenizer)
    
    #assemble all the scores
    scores = positive_scores - distribution_weights.reshape(-1,1)* negative_scores \
    - ( 1 - distribution_weights.reshape(-1,1))*C
    
    #calculate the scores from the discriminative model
    if discriminative_model is not None:
        discriminative_scores = bert_scorer(candidates1, candidates2, discriminative_model, tokenizer)
        discriminative_scores = (.5 - discriminative_scores )*1000
        scores += (distribution_weights > thresh)*discriminative_scores
    return scores