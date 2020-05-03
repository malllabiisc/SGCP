import os
import torch, pdb
import logging
from glob import glob
from torch.autograd import Variable
# import spacy

from nltk.translate.bleu_score import corpus_bleu
from src.bleu import compute_bleu
import numpy as np
import json
from gensim import models

#import ipdb as pdb

from collections import OrderedDict
from zss import simple_distance, Node
import subprocess
from src.word_embedding_utils import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
MULTI_BLEU_PERL = 'src/evaluation/multi-bleu.perl'

def run_multi_bleu(input_file, reference_file):
    bleu_output = subprocess.check_output(
        "./{} -lc {} < {}".format(MULTI_BLEU_PERL, reference_file, input_file),
        stderr=subprocess.STDOUT, shell=True).decode('utf-8')
    bleu = float(
        bleu_output.strip().split("\n")[-1]
        .split(",")[0].split("=")[1][1:])
    return bleu

class ContextFilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log.
    """
    def __init__(self, expt_name):
        super(ContextFilter, self).__init__()
        self.expt_name = expt_name

    def filter(self, record):
        record.expt_name = self.expt_name
        return True

def get_logger(name, expt_name, log_format, logging_level, log_file_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(log_format)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging_level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.addFilter(ContextFilter(expt_name))

    return logger


def print_log(logger, dict):
    str = ''
    for key, value in dict.items():
        str += '{}: {}\t'.format(key.replace('_', ' '), value)
    str = str.strip()
    logger.info(str)


def create_save_directories(log_fname, model_fname, run_name):
    log_folder_name = os.path.join(log_fname, run_name)

    if not os.path.exists(log_folder_name):
        os.makedirs(log_folder_name)

    if not os.path.exists(os.path.join(model_fname, run_name)):
        os.makedirs(os.path.join(model_fname, run_name))

    if not os.path.exists('tempgens'):
        os.makedirs('tempgens')

# Will save the checkpoint with the best validation scores (progressively), based on epoch number
def save_checkpoint(state, epoch, logger, path, args, old_path):
    sp          = path.split(".")

    file_path = '{}_{}.pth.tar'.format(sp[0].split("_")[0], epoch)
    logger.info('Saving checkpoint at : {}'.format(file_path))
    torch.save(state, file_path)

    if old_path!=None:
        old_ep = int(old_path.split("_")[-1].split(".")[0])
        if old_ep != 0 and os.path.exists(old_path):
            logger.info('Deleting old checkpoint at :{}'.format(old_path))
            os.remove(old_path)


# Will load the checkpoint based on the file_path provided
def load_checkpoint(model, mode, file_path, logger, device, pretrained_encoder=None):
    # cuda = torch.cuda.is_available()
    start_epoch = None
    train_loss = None
    val_loss = None
    voc = None
    bleu_score = None
    try:
        checkpoint = torch.load(file_path,
                                map_location=lambda storage,
                                loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer['tree'].load_state_dict(checkpoint['optimizer_tree_state_dict'])
        model.optimizer['rnn'].load_state_dict(checkpoint['optimizer_rnn_state_dict'])
        if pretrained_encoder is None:
            model.optimizer['encoder'].load_state_dict(checkpoint['optimizer_enc_state_dict'])
        start_epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        voc = checkpoint['voc']
        bleu_score = checkpoint['bleu']
        model.to(device)
        if mode == 'decode':
            model.eval()
        else:
            model.train()
        logger.info('Successfully loaded checkpoint from {}, with epoch number: {} for {}'.format(file_path, start_epoch, mode))
        return start_epoch, train_loss, val_loss, bleu_score, voc
    except:
        logger.info('No checkpoint found on {}'.format(file_path))
        return start_epoch, train_loss, val_loss, bleu_score, voc

# Will get the latest checkpoint based on model_dir and run_name. None if no checkpoint exists in the location
def get_latest_checkpoint(model_dir, run_name, logger, epoch=None):
    dir_name    = os.path.join(model_dir, run_name)
    ckpt_names  = glob('{}/*.pth.tar'.format(dir_name))
    ckpt_names = sorted(ckpt_names)
    checkpoint = None
    if len(ckpt_names) == 0:
        logger.info('No checkpoints found in dir_name {}'.format(dir_name))
    elif epoch is not None:
        checkpoint = ckpt_names[0]
        for ckpt_name in ckpt_names:
            path_tokens = ckpt_name.split('/')
            ep = path_tokens[-1].split(".")[0].split("_")[-1]
            if str(epoch) == ep:
                checkpoint = ckpt_name
                break

    else:
        latest_eps  = max([int(k.split("/")[-1].split(".")[0].split("_")[-1]) for k in ckpt_names])
        logger.info('Checkpoint found with epoch num {}'.format(latest_eps))
        checkpoint = ckpt_names[0]
        for ckpt_name in ckpt_names:
            path_tokens = ckpt_name.split('/')
            ep = path_tokens[-1].split(".")[0].split("_")[-1]
            if str(latest_eps) == ep:
                checkpoint = ckpt_name
                break

    return checkpoint

# For pytorch
def gpu_init_pytorch(gpu_num):
    torch.cuda.set_device(int(gpu_num))
    device = torch.device("cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu")

    return device


# Bleu Scorer (Send list of list of references, and a list of hypothesis)
def bleu_scorer(ref, hyp, script='default'):
    refsend = []
    for i in range(len(ref)):
        refsi = []
        for j in range(len(ref[i])):
            refsi.append(ref[i][j].split())
        refsend.append(refsi)

    gensend = []
    for i in range(len(hyp)):
        gensend.append(hyp[i].split())

    if script == 'nltk':
         metrics = corpus_bleu(refsend, gensend)
         return [metrics]

    metrics = compute_bleu(refsend, gensend)
    return metrics


def sort_pairs_by_src_len(src, tgt, tgt_dep_trees, tgt_pos):
    orig_idx = range(len(src))

    # Index by which sorting needs to be done
    sorted_idx = sorted(orig_idx, key=lambda k:len(src[k]), reverse=True)
    seq_pairs = list(zip(src, tgt, tgt_dep_trees, tgt_pos))
    seq_pairs = [seq_pairs[i] for i in sorted_idx]

    # For restoring original order
    orig_idx = sorted(orig_idx, key=lambda k:sorted_idx[k])
    src, tgt, tgt_dep_trees, tgt_pos = [s[0] for s in seq_pairs], [s[1] for s in seq_pairs], [s[2] for s in seq_pairs], [s[3] for s in seq_pairs]

    return src, tgt, tgt_dep_trees, tgt_pos, orig_idx


def pad_seq(seq, max_length, voc):
    seq += [voc.w2id['EOS'] for i in range(max_length - len(seq))]
    assert(len(seq) == max_length)
    return seq


def process_single(src, tgt, voc, device):
    src_len = len(src)
    tgt_len = len(tgt)
    src_padded = pad_seq(src, src_len, voc)
    tgt_padded = pad_seq(tgt, tgt_len, voc)

    src_var = Variable(torch.LongTensor(src_padded))
    tgt_var = Variable(torch.LongTensor(tgt_padded))

    src_var = src_var.to(device)
    tgt_var = tgt_var.to(device)

    return src_var, src_len, tgt_var, tgt_len

def process_batch(src, tgt, voc, device, tgt_dep_trees, tgt_pos):
    src, tgt, tgt_dep_trees, tgt_pos, orig_order    = sort_pairs_by_src_len(src, tgt, tgt_dep_trees, tgt_pos)
    src_len     = [len(s) for s in src]
    src_padded  = [pad_seq(s, max(src_len), voc) for s in src]
    tgt_len     = [len(t) for t in tgt]
    tgt_padded  = [pad_seq(t, max(tgt_len), voc) for t in tgt]
    tgt_pos_padded  = [pad_seq(t, max(tgt_len), voc) for t in tgt_pos]

    # Convert to max_len x batch_size
    src_var = Variable(torch.LongTensor(src_padded)).transpose(0, 1)
    tgt_var = Variable(torch.LongTensor(tgt_padded)).transpose(0, 1)
    tgt_pos_var = Variable(torch.LongTensor(tgt_pos_padded)).transpose(0, 1)

    src_var = src_var.to(device)
    tgt_var = tgt_var.to(device)
    tgt_pos_var = tgt_pos_var.to(device)

    adj = [[[], []] for i in range(45)]
    val = [[] for i in range(45)]
    offset = 0
    tgt_dep_trees = [json.loads(tree) for tree in tgt_dep_trees]

    for idx, tree in enumerate(tgt_dep_trees):
       for dep in tree:
         label = dep[2]
         if offset + dep[0] >= offset + tgt_len[idx] or offset + dep[1] >= offset + tgt_len[idx]:
             continue
         adj[label][0].append(offset + dep[0])
         adj[label][1].append(offset + dep[1])
         val[label].append(1)
       offset += tgt_len[idx]

    return src_var, src_len, tgt_var, tgt_pos_var, tgt_len, adj, val, orig_order


def get_sent_from_tree(tree, root, sent):
    if root not in tree:
        sent.append(root)
        return

    if len(tree[root]) == 0:
        sent.append("")
        return
    for child in tree[root]:
        get_sent_from_tree(tree, child, sent)

def get_sents_from_trees(trees):
    sents = []
    for tree in trees:
        if type(tree) == str:
            tree = json.loads(tree)
        sent = []
        get_sent_from_tree(tree, 'ROOT', sent)
        sents.append(sent)
    return sents

def get_sent_from_tree_bpe(tree, root, sent):
    if root not in tree:
        sent += root.split()
        return

    if len(tree[root]) == 0:
        sent.append("")
        return
    for child in tree[root]:
        get_sent_from_tree_bpe(tree, child, sent)

def get_sents_from_trees_bpe(trees):
    sents = []
    for tree in trees:
        if type(tree) == str:
            tree = json.loads(tree)
        sent = []
        get_sent_from_tree_bpe(tree, 'ROOT', sent)
        sents.append(sent)
    return sents

# Removes leaves of tree
def remove_leaves_from_tree(tree, root, bpe = False):
    mod_list = []
    for child in tree[root]:
        if child in tree:
            mod_list.append(child)
            remove_leaves_from_tree(tree, child, bpe)
    if mod_list == [] and bpe:
        mod_list = [len(tree[root][0].split())]
    tree[root] = mod_list

def remove_leaves_from_trees(trees, bpe = False):
    for tree in trees:
        remove_leaves_from_tree(tree, 'ROOT', bpe)


def trim_tree(tree, height, root, cur_height):
    if len(tree[root]) == 0:
        return [1]
    if type(tree[root][0]) == int:
        subword_len = tree[root][0]
        tree[root] = []
        return [subword_len]
    retlist = []
    if cur_height >= height:
        for child in tree[root]:
            retlist += trim_tree(tree, height, child, cur_height + 1)
            tree.pop(child)
        tree[root] = []
        return [np.sum(retlist)]
    else:
        for child in tree[root]:
            retlist += trim_tree(tree, height, child, cur_height + 1)
        return retlist
    # for child in tree[root]:
    #     trim_tree(tree, height, child, cur_height + 1)

def trim_trees(trees, height):
    phrase_ends = []
    trimmed_trees = []
    for tree in trees:
        tree = json.loads(tree)
        cur_phrase_lengths = trim_tree(tree, height, 'ROOT', 1)
        cur_phrase_ends = [0]
        for l in cur_phrase_lengths:
            cur_phrase_ends.append(cur_phrase_ends[-1] + l)
        cur_phrase_ends.pop()
        phrase_ends.append(cur_phrase_ends)
        trimmed_trees.append(tree)
    return trimmed_trees, phrase_ends

def bpestr2wordstr(bpestr):
    return bpestr.replace('@@ ', '')


def convert(args, tree, label, height):
    node = Node(label)
    if height == args.tree_height2:
        return (node, height)

    heights = [height]
    for child in tree[label]:
        if child in tree:
            kid, height_kid = convert(args, tree, child, height + 1)
            heights.append(height_kid)
            node.addkid(kid)
    return (node, max(heights))

def get_dist(args, s1, s2):
    try:
        t1, height_t1 = convert(args, json.loads(s1), 'ROOT', 1)
        t2, height_t2 = convert(args, json.loads(s2), 'ROOT', 1)
        return simple_distance(t1, t2)*(height_t1+height_t2)/(height_t1*height_t2)

    except:
        return None

def check_nan(model, name):
    for p in model.parameters():
        norm = p.norm()
        if torch.isnan(norm):
            pdb.set_trace()

def check_gradients(model, name):
    for i,p in enumerate(model.parameters()):
        if p.grad is not None:
            norm = p.grad.norm()
            if torch.isnan(norm):
                pdb.set_trace()
                return True
    return False

def initialize_embeddings(embedding, embed_path, embed_file, voc, device, vectype='glove'):

    '''
    Args:
        -embedding : embedding module
        -embed_file : file containing the pretrained embeddings
        -voc : Vocbulary object
    '''

    if vectype == 'glove':
        if not os.path.exists(os.path.join(embed_path, embed_file)):
            read_word_embedding(embed_path, embed_file[:-4], embed_file)
        else:
            pass
        word_embeddings = load_word_embedding(embed_path, embed_file)
        kw = list(word_embeddings.keys())[0]
    elif vectype == 'word2vec':
        word_embeddings = models.KeyedVectors.load_word2vec_format(os.path.join(embed_path,
                                                                                embed_file),
                                                                   limit=200000, binary=True)
        kw = 'the'
    else:
        print('Undefined vectype, Exiting!!')
        sys.exit()

    words = list(voc.w2id.keys())
    embed_dim = word_embeddings[kw].shape[-1]
    embedding_mat = np.zeros((len(words), embed_dim))

    for i, word in voc.id2w.items():

        if word == 'EOS':
            embedding_mat[i] = torch.zeros(embed_dim)
        elif word not in word_embeddings:
            embedding_mat[i] = np.random.randn(embed_dim)*np.sqrt(2/(len(words) + embed_dim))
        else:
            embedding_mat[i] = word_embeddings[word]

    embedding.weight.data = torch.FloatTensor(embedding_mat).to(device)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\?", " ? ", string)

    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
