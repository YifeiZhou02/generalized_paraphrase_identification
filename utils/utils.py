import pandas as pd
import torch
import os
from .question_pair import Question_pair, MAX_LEN, VOC_SIZE

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
def make_generative_data(pairs, option = 'positive', MAX_LEN = MAX_LEN):
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

def make_discriminative_data(pairs, MAX_LEN = MAX_LEN):
  """
  generate data for multitask training
  """
  features = []
  labels = []
  for pair in pairs:
    features.append(pair.tokens_combined.reshape(1,MAX_LEN*2))
    labels.append(pair.is_duplicate)
  return torch.cat(features, dim = 0), torch.Tensor(labels).long().reshape(-1,1)


def get_wmt17_seg_da(lang_pair):
    first, second = lang_pair.split("-")

    DA_scores = pd.read_csv(
        "wmt17/manual-evaluation/DA-seglevel.csv", delimiter=" ")
    DA_scores = DA_scores[DA_scores['LP'] == lang_pair]
    
    lang_dir = "wmt17/input/"\
        "wmt17-metrics-task/wmt17-submitted-data/"\
        "txt/system-outputs/newstest2017/{}".format(lang_pair)
    systems = [system[13:-6] for system in os.listdir(lang_dir)]
    sentences = {}

    with open("wmt17/input/wmt17-metrics-task/"
                "wmt17-submitted-data/txt/references/newstest2017-{}{}-ref.{}".format(first, second, second)) as f:
        references = f.read().strip().split("\n")
    for system in systems:
        with open("wmt17/input/"\
        "wmt17-metrics-task/wmt17-submitted-data/"\
        "txt/system-outputs/newstest2017/{}/newstest2017.{}.{}".format(lang_pair, system, lang_pair)) as f:
            sentences[system] = f.read().split("\n")

    gold_scores = []
    refs = []
    cands = []
    for index, row in DA_scores.iterrows():
        if not row['SYSTEM'] in systems:
            continue
        cands += [sentences[row['SYSTEM']][row['SID']-1]]
        refs += [references[row['SID']-1]]
        gold_scores += [row['HUMAN']]
    return refs, cands, gold_scores