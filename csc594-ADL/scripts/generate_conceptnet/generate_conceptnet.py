import pandas as pd
import numpy as np
import os.path
from pathlib import Path
from copy import deepcopy
import sys, re, csv
import json, requests
import torch, spacy, nltk
from torch import nn
from torch.utils.data import Dataset, DataLoader
from nltk import pos_tag
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
from pattern.en import conjugate, PARTICIPLE, referenced, INDEFINITE, pluralize

gpt2_model = 'gpt2'
template_repo = './templates/'
single_templates= 'relation_map.json'
multiple_templates = 'relation_map_multiple.json'
data_repo = './data/'
test_data = 'train600k.txt'
ofilename = "sentences_train600k"

class CommonsenseTuples(Dataset):
    """ Base class for generating sentences from relational triples """

    def __init__(self, tuple_dir, device, language_model=None, template_loc=None):
        """
        Args:
            tuple_dir (string): Path to the csv file with commonsense tuples
        """
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.device = device
        self.model = language_model
        if self.model is not None:
            self.model.eval()
            self.model.to(self.device)
        self.template_loc = template_loc

        # Load tuples
        with open(tuple_dir) as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            self.tuples = [row for row in reader]

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        # get tuple
        relation, head, tail = self.tuples[idx][:3]
        sent = self.apply_template(relation, head, tail)
        sent = f"{sent}."
        return sent

    def apply_template(self, relation, head, tail):
        """ To be overriden, returning the sentence, head, and tail """
        return


class EnumeratedTemplate(CommonsenseTuples):
    """ Sentence generation with coherency ranking """

    def __init__(self, *args, language_model=None, template_loc='./relation_map_multiple.json'):
        super().__init__(*args, language_model=language_model, template_loc=template_loc)
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        print("Loading template JSON.")
        with open(self.template_loc, 'r') as f:
            self.templates = json.load(f)

    def apply_template(self, relation, head, tail):
        candidate_sents = self.get_candidates(relation, head, tail)
        if candidate_sents:
            sent, head, tail = self.get_best_candidate(candidate_sents)
        else:
            sent = ""
        return sent

    def get_candidates(self, relation, head, tail):
        heads = self.formats(head)
        tails = self.formats(tail)
        candidate_sents = []
        try:
            templates = self.templates[relation]
        except:
            print("Error.")
            return candidate_sents
        for h in heads:
            for t in tails:
                for temp in templates:
                    candidate_sents.append((temp.format(h, t), h, t))
        return candidate_sents

    def formats(self, phrase):
        doc = self.nlp(phrase)
        first_word_POS = doc[0].pos_

        tokens = phrase.split(' ')
        new_tokens = tokens.copy()

        new_phrases = []
        # original
        new_phrases.append(' '.join(new_tokens))

        # with indefinite article
        if first_word_POS == 'NOUN' or first_word_POS == 'ADJ':
            new_tokens[0] = referenced(tokens[0])
            new_phrases.append(' '.join(new_tokens))
        # with definite article
        if first_word_POS == 'NOUN' or first_word_POS == 'ADJ':
            new_tokens[0] = "the " + tokens[0]
            new_phrases.append(' '.join(new_tokens))
        # as gerund
        if first_word_POS == 'VERB':
            new_tokens[0] = conjugate(tokens[0], tense=PARTICIPLE)
            new_phrases.append(' '.join(new_tokens))
            if len(tokens) > 1:
                if tokens[1] == 'to' and len(tokens) > 2:
                    new_tokens[2] = referenced(tokens[2])
                else:
                    new_tokens[1] = referenced(tokens[1])
            new_phrases.append(' '.join(new_tokens))
            new_tokens[0] = tokens[0]
            new_phrases.append(' '.join(new_tokens))

        # account for numbers
        if first_word_POS == 'NUM' and len(tokens) > 1:
            new_tokens[1] = pluralize(tokens[1])
            new_phrases.append(' '.join(new_tokens))
        return new_phrases

    def get_best_candidate(self, candidate_sents):
        candidate_sents.sort(key=self.score_sent, reverse=True)
        return candidate_sents[0]

    def score_sent(self, candidate):
        sent, _, _ = candidate
        sent = ". " + sent

        try:
            tokens = self.tokenizer.encode(sent)
        except KeyError:
            return 0

        context = torch.tensor([tokens], dtype=torch.long, device=self.device).reshape(1,-1)
        
        with torch.no_grad():
            logits, _ = self.model(context)
            
        log_probs = logits.log_softmax(2)
        sentence_log_prob = 0

        for idx, c in enumerate(tokens):
            if idx > 0:
                sentence_log_prob += log_probs[0, idx-1, c]

        return sentence_log_prob.item() / (len(tokens)**0.2)


class KnowledgeMiner:

    def __init__(self, dev_data_path, device, Template, **kwarg):
        """ Creates a class instance for doing KBC with a given template and
        HuggingFace model."""
        self.sentences = Template(dev_data_path, device, **kwarg)

    def make_predictions(self):
        data = []
        n = len(self.sentences)
        filepathname = os.path.join("../", f"{ofilename}.txt")
        for idx, sent in enumerate(self.sentences):
            data.append(sent)
            if idx > 0 and idx % 100 == 0:
                print("{}%".format(idx/n * 100))
                print("Writing sentences to file...")        
                with open(filepathname, "a") as outfile:
                    outfile.write("\n".join(data) + "\n")
                data = []

def run_experiment(template_type, knowledge_miners):
    print(f'Collecting {template_type} sentences...')
    ck_miner = knowledge_miners[template_type]
    sents = ck_miner.make_predictions()
    return sents

def mine(hardware):
    print('Loading GPT2...')
    gpt = GPT2LMHeadModel.from_pretrained(gpt2_model)

    knowledge_miners = {
        'coherency': KnowledgeMiner(
            data_repo + test_data,
            hardware,
            EnumeratedTemplate,
            language_model = gpt,
            template_loc = template_repo + multiple_templates)
    }

    return run_experiment('coherency', knowledge_miners)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python ckbc_experiments.py -<cuda or cpu>')
    else:
        hardware = sys.argv[1].replace('-', '', 1)
        mine(hardware)
        print("Complete!")