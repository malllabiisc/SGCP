from __future__ import print_function, division
import re, os, sys
import json
try:
    import cPickle as pickle
except ImportError:
    import pickle

import torch
# from torch_geometric.data import Data, DataLoader

import unicodedata
import matplotlib.pyplot as plt
import numpy as np

import ipdb as pdb
from torch.utils.data import Dataset, DataLoader, Sampler, IterableDataset
from collections import namedtuple, defaultdict
from itertools import chain, cycle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from src.helper import *
import h5py

class IterableParaphraseDataset(IterableDataset):
    def __init__(self, args, logger, datatype, is_train = False):
        logger.info("Creating Iterable Dataset")
        self.args = args
        self.is_train = is_train
        dataset = args.dataset

        self.orig_filename = os.path.join('data', dataset, datatype, 'src.txt')
        self.para_filename =  os.path.join('data', dataset, datatype, 'tgt.txt')

        logger.info("Done")

    def preprocess(self, x):
        if len(x[0]) == 0 or len(x[1]) == 0:
            pair = {
                'src' : None,
                'tgt' : None,
                'src_tree' : None,
                'tgt_tree' : None
            }
            return pair
        orig_tree, para_tree = x
        orig_tree, para_tree = json.loads(orig_tree), json.loads(para_tree)

        if not self.args.bpe:
            orig_sent = get_sents_from_trees([orig_tree])[0]
            para_sent = get_sents_from_trees([para_tree])[0]
        else:
            orig_sent = get_sents_from_trees_bpe([orig_tree])[0]
            para_sent = get_sents_from_trees_bpe([para_tree])[0]

        remove_leaves_from_trees([orig_tree], bpe=self.args.bpe)
        remove_leaves_from_trees([para_tree], bpe=self.args.bpe)

        if len(para_sent) > self.args.max_length or len(para_sent) == 0:
            pair = {
                'src' : None,
                'tgt' : None,
                'src_tree' : None,
                'tgt_tree' : None
            }
            return pair

        orig_tree = json.dumps(orig_tree)
        para_tree = json.dumps(para_tree)
        orig_sent = json.dumps(orig_sent)
        para_sent = json.dumps(para_sent)

        pair = {
                'src': orig_sent,
                'tgt': para_sent,
                'src_tree': orig_tree,
                'tgt_tree': para_tree,
                'src_embed': " "
            }
        return pair

    def __iter__(self):
        self.orig_file = open('{}-corenlp-opti'.format(self.orig_filename), 'r', encoding='utf-8', errors='ignore')
        self.para_file = open('{}-corenlp-opti'.format(self.para_filename), 'r', encoding='utf-8', errors='ignore')

        if self.is_train and not self.args.single_side:
            iter1 = map(self.preprocess, zip(self.orig_file, self.para_file))
            self.orig_file = open('{}-corenlp-opti'.format(self.orig_filename), 'r', encoding='utf-8', errors='ignore')
            self.para_file = open('{}-corenlp-opti'.format(self.para_filename), 'r', encoding='utf-8', errors='ignore')
            iter2 = map(self.preprocess, zip(self.para_file, self.orig_file))
            return chain(iter1, iter2)
        else:
            iter1 = map(self.preprocess, zip(self.orig_file, self.para_file))
            return iter1

class IterableValParaphraseDataset(IterableDataset):
    def __init__(self, args, logger, datatype, is_train = False):
        logger.info("Creating Iterable Dataset")
        self.args = args
        self.is_train = is_train
        dataset = args.dataset

        self.orig_filename = os.path.join('data', dataset, datatype, 'src.txt')
        self.para_filename  =  os.path.join('data', dataset, datatype, 'ref.txt')
        self.template_filename = os.path.join('data',dataset, datatype, 'tgt.txt')

        logger.info("Done")

    def preprocess(self, x):
        if len(x[0]) == 0 or len(x[2]) == 0:
            pair = {
                'src' : None,
                'tgt' : None,
                'temp' : None,
                'src_tree' : None,
                'tgt_tree' : None
            }
            return pair

        orig_tree, para_tree, template = x
        orig_tree, para_tree, template = json.loads(orig_tree), json.loads(para_tree), json.loads(template)

        if not self.args.bpe:
            orig_sent = get_sents_from_trees([orig_tree])[0]
            para_sent = get_sents_from_trees([para_tree])[0]
            temp_sent = get_sents_from_trees([template])[0]

        else:
            orig_sent = get_sents_from_trees_bpe([orig_tree])[0]
            para_sent = get_sents_from_trees_bpe([para_tree])[0]
            temp_sent = get_sents_from_trees_bpe([template])[0]

        remove_leaves_from_trees([orig_tree], bpe = self.args.bpe)
        remove_leaves_from_trees([para_tree], bpe = self.args.bpe)
        remove_leaves_from_trees([template], bpe = self.args.bpe)

        if len(para_sent) > self.args.max_length or len(para_sent) == 0:
            pair = {
                'src' : None,
                'tgt' : None,
                'temp' : None,
                'src_tree' : None,
                'tgt_tree' : None
            }
            return pair

        orig_tree = json.dumps(orig_tree)
        para_tree = json.dumps(para_tree)
        template =  json.dumps(template)
        orig_sent = json.dumps(orig_sent)
        para_sent = json.dumps(para_sent)
        temp_sent = json.dumps(temp_sent)

        pair = {
                'src': orig_sent,
                'tgt': para_sent,
                'temp': temp_sent,
                'src_tree': orig_tree,
                'tgt_tree': template,
                'src_embed': " "
            }

        return pair

    def __iter__(self):
        self.orig_file = open('{}-corenlp-opti'.format(self.orig_filename), 'r', encoding='utf-8', errors='ignore')
        self.para_file = open('{}-corenlp-opti'.format(self.para_filename), 'r', encoding='utf-8', errors='ignore')
        self.template_file = open('{}-corenlp-opti'.format(self.template_filename), 'r', encoding = 'utf-8', errors = 'ignore')

        iter1 = map(self.preprocess, zip(self.orig_file, self.para_file, self.template_file))

        return iter1

class Voc:
    def __init__(self, name):
        self.name       = name
        self.trimmed    = False
        self.frequented = False
        self.w2id       = {'SOS': 0, 'EOS': 1, 'UNK': 2}
        self.id2w       = {0: 'SOS', 1: 'EOS', 2: 'UNK'}
        self.w2c        = defaultdict(int)
        self.nwords     = 3

    def addSentence(self, sent):
        for word in sent:
            self.addWord(word)

    def addTree(self, tree, root):
        if root not in tree:
            self.addWord(root)
            return
        else:
            self.addWord(root.split('-')[0])
        for child in tree[root]:
            self.addTree(tree, child)

    def addWord(self, word):
        if word not in self.w2id:
            self.w2id[word]     = self.nwords
            self.id2w[self.nwords]   = word
            self.w2c[word]      = 1
            self.nwords         = self.nwords + 1
        else:
            self.w2c[word]      = self.w2c[word] + 1

    def trim(self, mincount):
        if self.trimmed == True:
            return
        self.trimmed    = True

        keep_words = []
        for k, v in self.w2c.items():
            if v >= mincount:
                keep_words += [k]*v

        self.w2id       = {'SOS': 0, 'EOS': 1, 'UNK': 2}
        self.id2w       = {0: 'SOS', 1: 'EOS', 2: 'UNK'}
        self.w2c        = defaultdict(int)
        self.nwords     = 3
        for word in keep_words:
            self.addWord(word)

    def most_frequent(self, topk):
        if self.frequented == True:
            return
        self.frequented     = True

        keep_words = []
        count      = 3
        sorted_by_value = sorted(self.w2c.items(), key=lambda kv: kv[1], reverse=True)
        for word, freq in sorted_by_value:
            keep_words  += [word]*freq
            count += 1
            if count == topk:
                break

        self.w2id       = {'SOS': 0, 'EOS': 1, 'UNK': 2}
        self.id2w       = {0: 'SOS', 1: 'EOS', 2: 'UNK'}
        self.w2c        = defaultdict(int)
        self.nwords     = 3
        for word in keep_words:
            self.addWord(word)

def sent2id(voc, sent, max_length, is_ptr=False):
    idx_vec = []
    idx_extend_vec = []
    oov = []
    for w in sent:
        if w in voc.w2id:
            idx = voc.w2id[w]
            idx_vec.append(idx)
            idx_extend_vec.append(idx)
        else:
            if is_ptr:
                oov.append(w)
                idx_extend_vec.append(voc.nwords + oov.index(w))
                idx_vec.append(voc.w2id['UNK'])
            else:
                idx_vec.append(voc.w2id['UNK'])
                idx_extend_vec.append(voc.w2id['UNK'])
    return idx_vec[:max_length], idx_extend_vec[:max_length], oov


def sents2ids(voc, sents, max_length, is_ptr=False):
    all_ids = []
    all_extend_ids = []
    oovs = []
    for sent in sents:
        sent = json.loads(sent)
        ids, extend_ids, oov = sent2id(voc, sent, max_length, is_ptr)
        all_ids.append(ids)
        all_extend_ids.append(extend_ids)
        oovs.append(oov)
    return all_ids, all_extend_ids, oovs

def para2id(voc, para, max_length=20, sent_oov=None):
    para_inp = [voc.w2id['SOS']]
    para_tgt = []
    for w in para:
        if w in voc.w2id:
            idx = voc.w2id[w]
            para_inp.append(idx)
            para_tgt.append(idx)
        else:
            if sent_oov is None:
                para_inp.append(voc.w2id['UNK'])
                para_tgt.append(voc.w2id['UNK'])
            else:
                if w in sent_oov:
                    para_inp.append(voc.w2id['UNK'])
                    para_tgt.append(voc.nwords + sent_oov.index(w))
                else:
                    para_inp.append(voc.w2id['UNK'])
                    para_tgt.append(voc.w2id['UNK'])

    para_tgt.append(voc.w2id['EOS'])

    assert len(para_tgt) == len(para_inp)
    return para_inp[:max_length], para_tgt[:max_length]


def paras2ids(voc, paras, max_length=20, sent_oovs=None):
    all_inp_ids = []
    all_tgt_ids = []
    for i, para in enumerate(paras):
        para = json.loads(para)
        if sent_oovs is not None:
            ids_inp, ids_tgt = para2id(voc, para, max_length, sent_oovs[i])
        else:
            ids_inp, ids_tgt = para2id(voc, para, max_length, None)
        all_inp_ids.append(ids_inp)
        all_tgt_ids.append(ids_tgt)
    return all_inp_ids, all_tgt_ids



def id2sent(voc, oov, tensor, no_eos=False):
    sentence_word_list =  []
    for idx in tensor:
        if idx.item() >= voc.nwords:
            w = oov[idx.item()-voc.nwords]
        else:
            w = voc.id2w[idx.item()]

        if w == 'EOS':
            break
        sentence_word_list.append(w)
    sentence = ' '.join(sentence_word_list)
    sentence = bpestr2wordstr(sentence)
    return sentence.split()


def ids2sents(voc, oovs, tensors, no_eos=False):
    tensors = tensors.transpose(0, 1)
    batch_word_list = []
    for i, tens in enumerate(tensors):
       batch_word_list.append(id2sent(voc, oovs[i], tens, no_eos))
    return batch_word_list

def lsid2sent(voc, oov, ls, no_eos=False):
    sentence_word_list =  []
    for idx in ls:
        if idx >= voc.nwords:
            w = oov[idx-voc.nwords]
        else:
            w = voc.id2w[idx]

        if w == 'EOS':
            break
        sentence_word_list.append(w)

    sentence = ' '.join(sentence_word_list)
    sentence = bpestr2wordstr(sentence)
    return sentence.split()

def lsids2sents(voc, oovs, ls, no_eos = False):
    batch_word_list = []
    for i, tens in enumerate(ls):
       batch_word_list.append(lsid2sent(voc, oovs[i], tens, no_eos))
    return batch_word_list

def collate(pairs):

    srcs = [pairs[i]['src'] for i in range(len(pairs)) if pairs[i]['src'] is not None]
    tgts = [pairs[i]['tgt'] for i in range(len(pairs)) if pairs[i]['tgt'] is not None]
    src_trees = [pairs[i]['src_tree'] for i in range(len(pairs)) if pairs[i]['src_tree'] is not None]
    tgt_trees = [pairs[i]['tgt_tree'] for i in range(len(pairs)) if pairs[i]['tgt_tree'] is not None]

    return {
        'src' : srcs,
        'tgt' : tgts,
        'src_tree' : src_trees,
        'tgt_tree' : tgt_trees,
        'src_embed' : " "
    }

def collateval(pairs):
    srcs = [pairs[i]['src'] for i in range(len(pairs)) if pairs[i]['src'] is not None]
    tgts = [pairs[i]['tgt'] for i in range(len(pairs)) if pairs[i]['tgt'] is not None]
    temps = [pairs[i]['temp'] for i in range(len(pairs)) if pairs[i]['temp'] is not None]
    src_trees = [pairs[i]['src_tree'] for i in range(len(pairs)) if pairs[i]['src_tree'] is not None]
    tgt_trees = [pairs[i]['tgt_tree'] for i in range(len(pairs)) if pairs[i]['tgt_tree'] is not None]

    return {
        'src' : srcs,
        'tgt' : tgts,
        'temp' : temps,
        'src_tree' : src_trees,
        'tgt_tree' : tgt_trees,
        'src_embed' : " "
    }
