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

<code>python run.py --source_dataset [QQP, PIT, PAWS] --</code>
