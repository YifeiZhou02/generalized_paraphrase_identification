# generalized_paraphrase_identification
Implementation of the paper **'GAPX: Generalized Autoregressive Paraphrase-identification X'** <br />

**NeurIPS 2022**

An ensemble model for paraphrase identification robust to distribution shift.

![Canvas 1](https://user-images.githubusercontent.com/83000332/192064352-84d1ac9b-14d7-4697-96f2-49347c7b44d4.png)


## Requirements
* GPU
* requirements.txt

## Dataset 
Please download the following paraphrase identification datasets:
* Quora Question Pair : https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs
* Paraphrase and Semantic Similarity on Twitter: https://github.com/cocoxu/SemEval-PIT2015
* Paraphrase Adversarials from Word Scrambling (en): https://www.statmt.org/wmt17/metrics-task.html
* WMT 17: https://www.statmt.org/wmt17/metrics-task.html

## Usage
To train and evaluate a paraphrase identification model, run: <br />  

<code>python run.py --source_dataset [QQP, PIT, PAWS] --option [naive, robust]</code>
<br />  

Here we implemented a simplified version from the paper, where for the discriminative model, we use BART instead of RoBERTa

## Results
You should expect to see something similar to this (f1/acc/auc):
| Command    |  QQP->QQP     |QQP->WMT     |QQP->PAWS     |QQP->PIT      |
| :---        |    :----:   |          ---: |      ---: |      ---: |
|   <code>python run.py --source_dataset QQP --option naive</code>     |    83.4/83.5/91.2  | 66.7/66.8/74.2 | 44.7/49.8/57.1 | 63.6/66.5/82.0      |
| <code>python run.py --source_dataset QQP --option robust</code>       |   83.1/83.2/88.4   | 74.4/74.7/79.3| 56.6/56.9/59.5 | 62.3/63.6/73.5      |
