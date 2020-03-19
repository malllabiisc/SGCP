import torch
from torch.utils.data import Dataset, IterableDataset
import pandas as pd
from pytorch_transformers import RobertaTokenizer
from src.helper import *
from itertools import chain, cycle
import pdb

class ParaphraseDataset(Dataset):
    def __init__(self, filename, maxlen = 30):
        self.df = pd.read_csv(filename)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        q1 = str(self.df.loc[index,'sentence1']).lower()
        q2 = str(self.df.loc[index,'sentence2']).lower()
        
        label = self.df.loc[index, 'label']
        return q1, q2, label

class IterableParaphraseDataset(IterableDataset):
    def __init__(self, args, logger, datatype, is_train = False):
        #logger.info("Creating Iterable Dataset")
        self.args = args
        self.is_train = is_train
        dataset = args.dataset

        self.orig_filename = os.path.join('data', dataset, '{}_src.txt'.format(datatype))
        self.para_filename =  os.path.join('data', dataset, '{}_tgt.txt'.format(datatype))

        #logger.info("Done")

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

        orig_sent = get_sents_from_trees([orig_tree])[0]
        para_sent = get_sents_from_trees([para_tree])[0]
        orig_sent = json.dumps(orig_sent)
        para_sent = json.dumps(para_sent)
        orig_sent = orig_sent[1:-1].replace("\"", "").replace(",","").split()
        para_sent = para_sent[1:-1].replace("\"", "").replace(",","").split()
        return " ".join(orig_sent), " ".join(para_sent), 1

    def __iter__(self):
        self.orig_file = open('{}-corenlp-opti'.format(self.orig_filename), 'r', encoding='utf-8', errors='ignore')
        self.para_file = open('{}-corenlp-opti'.format(self.para_filename), 'r', encoding='utf-8', errors='ignore')

        iter1 = map(self.preprocess, zip(self.orig_file, self.para_file))
        iter2 = map(self.preprocess, zip(self.para_file, self.orig_file))

        return chain(iter1, iter2)