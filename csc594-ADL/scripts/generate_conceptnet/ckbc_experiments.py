import sys
import csv

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from sentences import EnumeratedTemplate
from knowledge_miner import KnowledgeMiner
from pytorch_pretrained_bert import BertForMaskedLM, GPT2LMHeadModel

bert_model = 'bert-large-uncased'
gpt2_model = 'gpt2'

template_repo = './templates/'
single_templates= 'relation_map.json'
multiple_templates = 'relation_map_multiple.json'

data_repo = './data/'
test_data = 'valid.txt'

def run_experiment(template_type, knowledge_miners):
    print(f'Collecting {template_type} sentences...')
    ck_miner = knowledge_miners[template_type]
    sents = ck_miner.make_predictions()
    return sents

def mine(hardware):
    print('loading BERT...')
    bert = BertForMaskedLM.from_pretrained(bert_model)
    print('loading GPT2...')
    gpt = GPT2LMHeadModel.from_pretrained(gpt2_model)

    knowledge_miners = {
        'coherency': KnowledgeMiner(
            data_repo + test_data,
            hardware,
            EnumeratedTemplate,
            bert,
            language_model = gpt,
            template_loc = template_repo + multiple_templates
        )
    }

    return run_experiment('coherency', knowledge_miners)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python ckbc_experiments.py -<cuda or cpu>')
    else:
        hardware = sys.argv[1].replace('-', '', 1)
        sents = mine(hardware)
        print("Complete!")