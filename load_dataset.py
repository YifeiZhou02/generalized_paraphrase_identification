import pandas as pd
import os
import torch
def get_wmt18_seg_data(lang_pair):
    src, tgt = lang_pair.split('-')
    
    RRdata = pd.read_csv(
        "wmt18/wmt18/wmt18-metrics-task-package/manual-evaluation/RR-seglevel.csv", sep=' ')
    # RRdata_lang = RRdata[RRdata['LP'] == lang_pair] # there is a typo in this data. One column name is missing in the header
    RRdata_lang = RRdata[RRdata.index == lang_pair]

    systems = set(RRdata_lang['BETTER'])
    systems.update(list(set(RRdata_lang['WORSE'])))
    systems = list(systems)
    sentences = {}
    for system in systems:
        with open("wmt18/wmt18/wmt18-metrics-task-package/input/wmt18-metrics-task-nohybrids/system-outputs/newstest2018/{}/newstest2018.{}.{}".format(lang_pair, system, lang_pair)) as f:
            sentences[system] = f.read().split("\n")

    with open("wmt18/wmt18/wmt18-metrics-task-package/input/wmt18-metrics-task-nohybrids/"
              "references/{}".format('newstest2018-{}{}-ref.{}'.format(src, tgt, tgt))) as f:
        references = f.read().split("\n")

    ref = []
    cand_better = []
    cand_worse = []
    for index, row in RRdata_lang.iterrows():
        cand_better += [sentences[row['BETTER']][row['SID']-1]]
        cand_worse += [sentences[row['WORSE']][row['SID']-1]]
        ref += [references[row['SID']-1]]

    return ref, cand_better, cand_worse


def get_wmt17_seg_data(lang_pair):
    src, tgt = lang_pair.split('-')
    
    RRdata = pd.read_csv(
        "wmt17/manual-evaluation/RR-seglevel.csv", sep=' ')
    # RRdata_lang = RRdata[RRdata['LP'] == lang_pair] # there is a typo in this data. One column name is missing in the header
    RRdata_lang = RRdata[RRdata.index == lang_pair]

    systems = set(RRdata_lang['BETTER'])
    systems.update(list(set(RRdata_lang['WORSE'])))
    systems = list(systems)
    sentences = {}
    for system in systems:
        with open("wmt17/input/"\
        "wmt17-metrics-task/wmt17-submitted-data/"\
        "txt/system-outputs/newstest2017/{}/newstest2017.{}.{}".format(lang_pair, system, lang_pair)) as f:
            sentences[system] = f.read().split("\n")

    with open("wmt17/input/wmt17-metrics-task/"
                "wmt17-submitted-data/txt/references/newstest2017-{}{}-ref.{}".format(src, tgt, tgt)) as f:
        references = f.read().split("\n")

    ref = []
    cand_better = []
    cand_worse = []
    for index, row in RRdata_lang.iterrows():
        cand_better += [sentences[row['BETTER']][row['SID']-1]]
        cand_worse += [sentences[row['WORSE']][row['SID']-1]]
        ref += [references[row['SID']-1]]
    return ref, cand_better, cand_worse
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

def kendell_score(scores_better, scores_worse):
    total = len(scores_better)
    correct = torch.sum(scores_better > scores_worse).item()
    incorrect = total - correct
    return (correct - incorrect)/total