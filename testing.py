import torch
import torch.nn as nn
from sklearn import metrics
from scipy.stats import exponweib
from utils.question_pair import MAX_LEN, VOC_SIZE

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
  """
  This function assign a average of the mean cross entropy of
  1) generating ref conditioned on cand
  2) generating cand conditioned on ref
  """
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
  use the naive model (basic bert or roberta model) to assign scores to each cand-ref pair
  """
  model.eval()
  model = model.to('cuda')
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
  """
  Returns the cross_entropy measure by the model of 
  generating (cand concat ref) without a condition
  """
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

def robust_predictions(positive_model, negative_model, distribution_model,\
                       dev_pairs, test_pairs, tokenizer, discriminative_model= None, C = 3):
    """
    this method assembles all the models to make robust predictions for paraphrase identification
    the outputs are numerical scores, to make classification results, compare the scores with 0,
    >0 will be paraphrases
    """
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

