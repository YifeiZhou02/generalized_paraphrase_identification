import numpy as np
import torch
import random
from utils import Question_pair, get_pairs, get_pit_test_pairs, MAX_LEN, get_wmt17_seg_da
from training import get_generative_model, get_discriminative_model,  get_distribution_model
from testing import pairs_auc, robust_predictions, bert_scorer
from transformers import BartTokenizer
from transformers.utils import logging
wmt17_lang_pairs = ['cs-en', 'de-en', 'fi-en', 'lv-en', 'ru-en', 'tr-en', 'zh-en']
from sklearn.metrics import f1_score
import argparse

def main(args):
    logging.set_verbosity_error()
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    #load dataset
    all_refs = []
    all_cands = []
    all_gold_scores = []
    for lang_pair in wmt17_lang_pairs:
        refs, cands, gold_scores = get_wmt17_seg_da(lang_pair)
    for ref, cand, gold_score in zip(refs, cands, gold_scores):
        all_refs.append(ref)
        all_cands.append(cand)
        all_gold_scores.append(gold_score)
    wmt_test_pairs = []
    duplicate_counts = 0
    for ref, cand, gold_score in zip(all_refs, all_cands, all_gold_scores):
        if gold_score > 0:
            duplicate_counts += 1
            is_duplicate = 1
        else:
            is_duplicate = 0
        wmt_test_pairs.append(Question_pair(0, ref, cand,\
                                    is_duplicate))
    for pair in wmt_test_pairs:
        pair.tokenize_self(tokenizer, MAX_LEN)

    #get pit datasets
    pit_train_pairs = get_pairs('pit/train.data',
                                0, 2, 3, 4, tokenizer)
    pit_dev_pairs = get_pairs('pit/dev.data',
                                0, 2, 3, 4, tokenizer)
    pit_test_pairs = get_pit_test_pairs(tokenizer)
    random.seed(42)
    pit_train_positive = [p for p in pit_train_pairs if p.is_duplicate ]
    pit_train_negative = [p for p in pit_train_pairs if not p.is_duplicate ]
    pit_train_negative = random.sample(pit_train_negative, len(pit_train_positive))
    pit_train_pairs = pit_train_positive + pit_train_negative
    pit_dev_positive = [p for p in pit_dev_pairs if p.is_duplicate ]
    pit_dev_negative = [p for p in pit_dev_pairs if not p.is_duplicate ]
    pit_dev_negative = random.sample(pit_dev_negative, len(pit_dev_positive))
    pit_dev_pairs = pit_dev_positive + pit_dev_negative
    pit_test_positive = [p for p in pit_test_pairs if p.is_duplicate ]
    pit_test_negative = [p for p in pit_test_pairs if not p.is_duplicate ]
    pit_test_negative = random.sample(pit_test_negative, len(pit_test_positive))
    pit_test_pairs = pit_test_positive + pit_test_negative
    pit_test_pairs, pit_dev_pairs = pit_dev_pairs, pit_test_pairs
    random.shuffle(pit_train_pairs)
    random.seed(1024)

    #get paws dataset
    paws_test_pairs = get_pairs('paws/test_2k.tsv',
                            0, 1, 2, 3, tokenizer)
    #get qqp test
    qqp_test_pairs = get_pairs('qqp/test.tsv', 3,1,2,0, tokenizer)[:2000]


    empty = tokenizer('', max_length = MAX_LEN*2,
            truncation = True, padding = "max_length",
            add_special_tokens = True, return_token_type_ids = False,\
            return_attention_mask = False, return_tensors = 'pt')['input_ids']
    
    train_pairs = pit_train_pairs
    max_epochs = 5
    if args.source_dataset == "QQP":
        train_pairs = get_pairs('qqp/train.tsv', 3,1,2,0, tokenizer)[:10000]
        dev_pairs = get_pairs('qqp/dev.tsv', 3,1,2,0, tokenizer)[:2000]
    elif args.source_dataset == 'PIT':
        train_pairs = pit_train_pairs
        dev_pairs = pit_dev_pairs
        max_epochs = 10
    else:
        train_pairs = get_pairs('paws/train.tsv',
                            0, 1, 2, 3, tokenizer)
        dev_pairs = get_pairs('paws/dev_2k.tsv',
                            0, 1, 2, 3, tokenizer)
    #training the models
    discriminative_model = get_discriminative_model(train_pairs, dev_pairs, max_epochs = max_epochs)
    if args.option == 'robust':
        distribution_model = get_distribution_model(train_pairs, dev_pairs, empty, max_epochs = max_epochs)
        positive_model = get_generative_model('positive', train_pairs, dev_pairs, max_epochs = max_epochs)
        negative_model = get_generative_model('negative', train_pairs, dev_pairs, max_epochs = max_epochs)

    #making predictions
    for test_name, test_pairs, C in zip(['PIT', 'QQP', 'WMT', 'PAWS'], [pit_test_pairs, qqp_test_pairs,\
        wmt_test_pairs, paws_test_pairs], [5, 3, 3, 1]):
        if args.option == 'robust':
            scores = robust_predictions(positive_model, negative_model, distribution_model,\
                            dev_pairs, test_pairs, tokenizer, discriminative_model, C)
        else:
            scores = .5 - bert_scorer([pair.question1 for pair in test_pairs],\
                 [pair.question2 for pair in test_pairs], discriminative_model, tokenizer)
        scores = np.asarray(scores)

        #evaluate results
        auc_score = pairs_auc(test_pairs, scores)
        target = np.array([p.is_duplicate for p in test_pairs])
        predictions = scores.flatten() < 0
        f1 = f1_score(scores.flatten() < 0, target, average = 'macro')
        acc_score = np.mean(target == predictions)
        print(f'{args.source_dataset} -> {test_name}')
        print('f1/acc/auc')
        print("{:.1f}/{:.1f}/{:.1f}".format(f1*100, acc_score*100, auc_score*100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset', default="QQP", choices= ['QQP', 'PIT', 'PAWS'], type=str)
    parser.add_argument('--option', default="robust", choices= ["robust", "naive"], type=str)
    args = parser.parse_args()
    print(args.option)
    main(args)