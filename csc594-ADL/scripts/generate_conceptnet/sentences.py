"""
Collection of classes for use in generating sentences from relational triples
"""

from torch.utils.data import Dataset, DataLoader
from torch import nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, \
                                    GPT2LMHeadModel, GPT2Tokenizer
import torch
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from nltk import pos_tag
import nltk
from pathlib import Path
import spacy
# pip install pattern should work
from pattern.en import conjugate, PARTICIPLE, referenced, INDEFINITE, pluralize


class CommonsenseTuples(Dataset):
    """ Base class for generating sentences from relational triples """

    def __init__(self, tuple_dir, device, language_model=None, template_loc=None):
        """
        Args:
            tuple_dir (string): Path to the csv file with commonsense tuples
        """
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.sep_token = '[SEP]'
        self.start_token = '[CLS]'
        self.pad_token = '[PAD]'

        self.max_len = 20
        self.stop_tokens = ['the', 'a', 'an']
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
        r, t1, t2 = self.tuples[idx][:3]

        # apply template
        #try:
        sent, t1, t2 = self.apply_template(r, t1, t2)
        #except (json.JSONDecodeError) as e:
        #    return (-1,-1,-1,-1)
        # apply start and end tokens
        sent = f"{sent}."

        # tokenize sentences and t1 and t2
        tokenized_sent = self.tokenizer.tokenize(sent)

        # convert tokens to ids
        indexed_sent = self.tokenizer.convert_tokens_to_ids(tokenized_sent)

        return (
            torch.tensor(indexed_sent, device=self.device))


    def id_to_text(self, sent):
        if type(sent) == torch.Tensor:
            tokens = [self.tokenizer.ids_to_tokens[sent[idx].item()] for idx in range(len(sent))]
        else:
            tokens = [self.tokenizer.ids_to_tokens[sent[idx]] for idx in range(len(sent))]
        return " ".join(tokens)

    def apply_template(self, relation, head, tail):
        """ To be overriden, returning the sentence, head, and tail """
        return


class EnumeratedTemplate(CommonsenseTuples):
    """ Sentence generation with coherency ranking """

    def __init__(self, *args, language_model=None, template_loc='./relation_map_multiple.json'):
        super().__init__(*args, language_model=language_model, template_loc=template_loc)
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.enc = GPT2Tokenizer.from_pretrained('gpt2')
        print("Loading template JSON.")
        with open(self.template_loc, 'r') as f:
            self.templates = json.load(f)

    def apply_template(self, relation, head, tail):
        candidate_sents = self.get_candidates(relation, head, tail)
        sent, head, tail = self.get_best_candidate(candidate_sents)
        return sent, head, tail

    def get_candidates(self, relation, head, tail):
        heads = self.formats(head)
        tails = self.formats(tail)
        templates = self.templates[relation]
        candidate_sents = []
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
            new_tokens[0] = "the "+tokens[0]
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
        sent = ". "+sent

        try:
            tokens = self.enc.encode(sent)
        except KeyError:
            return 0

        context = torch.tensor(tokens, dtype=torch.long, device=self.device).reshape(1,-1)
        logits, _ = self.model(context)
        log_probs = logits.log_softmax(2)
        sentence_log_prob = 0

        for idx, c in enumerate(tokens):
            if idx > 0:
                sentence_log_prob += log_probs[0, idx-1, c]

        return sentence_log_prob.item() / (len(tokens)**0.2)
