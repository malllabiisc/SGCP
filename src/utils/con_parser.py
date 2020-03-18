import sys
from nltk.parse.corenlp import CoreNLPParser
from nltk.tree import Tree
import pdb as pdb
import argparse
import re
import json
from tqdm import tqdm
from time import time
from subword_nmt import apply_bpe

import multiprocessing as mp
import pickle
from itertools import repeat, count

# AVAILABLE SERVERS
SERVERS = {
    "localhost" : "0.0.0.0"
}

error_idx = open('data/error_idx.txt', 'w+', encoding='utf-8', errors='ignore')

def bpe_encode(word):
    if word in word2bpe:
        return word2bpe[word]

    bpe_segments = bpe_encoder.segment(word)
    word2bpe[word] = bpe_segments
    return bpe_segments

def parseradd(host, ports):
    parser = []
    for port in ports:
        parser.append(CoreNLPParser(url='http://{}:{}'.format(host, port)))
    return parser

def cleanbrackets(string):
    a = {
        '-LRB-': '(',
        '-RRB-': ')',
        '-LSB-': '[',
        '-RSB-': ']',
        '-LCB-': '{',
        '-RCB-': '}',
    }
    try:
        return a[string]
    except:
        return string

def clean_str(string):
    string = string.replace('-LRB-', ' ( ')
    string = string.replace('-RRB-', ' ) ')
    string = string.replace('-LSB-', ' [ ')
    string = string.replace('-RSB-', ' ] ')
    string = string.replace('-LCB-', ' { ')
    string = string.replace('-RCB-', ' } ')
    return string

def dfs(root, s):
    label = root._label.split()[0].replace('-','')
    #label = re.findall(r'-+[A-Z]+-|[A-Z]+\$|[A-Z]+|\.', root._label)[0]
    tree_dict = {label : []}
    # print(leaf, root._label)
    if len(root._label.split()) > 1:
        tree_dict[label].append(root._label.split()[1])
    for child in root:
        if type(child) is str:
            tree_dict[label] = bpe_encode(cleanbrackets(child))
            return tree_dict
        #if child._label.split()[]
        else:
            tree_dict[label].append(dfs(child, s))
    return tree_dict

def dfsopti(tree, root, tree_dict, label, cnt):
    tree_dict[label] = []

    if type(tree[root]) is str:
        tree_dict[label].append(tree[root])
    else:
        for child in tree[root]:
            if type(child) is dict:
                child_label = list(child.keys())[0]
                if child_label not in cnt:
                    cnt[child_label] = 0
                cnt[child_label] += 1
                child_label = child_label + '-' + str(cnt[child_label])
                tree_dict[label].append(child_label)
                dfsopti(child, list(child.keys())[0], tree_dict, child_label, cnt)

def parseget(i, line, parser):
    line = clean_str(line).strip()
    lenpar = len(parser)
    try:
        parses = parser[i%lenpar].parse(line.split())
    except:
        error_idx.write(str(i) + '\n')
        print('ERROR IN index {}'.format(i))
        return ''
    k = list(parses)
    tree = str(k[0]).replace("\n", "")
    tree = Tree.fromstring(tree)
    tree_dict = dfs(tree, line,)
    #pdb.set_trace()
    # print(tree_dict)
    tree = tree_dict
    tree_dict = {}
    cnt = {'ROOT': 0}
    dfsopti(tree, 'ROOT', tree_dict, 'ROOT', cnt)
    ans = json.dumps(tree_dict)
    print('Completed : {}'.format(i+1), end='\r')
    # outfile_corenlp.flush()
    return ans

def main(args):
    infile = open(args.infile, 'r', encoding='utf-8', errors='ignore')
    outfile_corenlp = open(args.infile + "-corenlp-opti", "w", buffering=-1, encoding='utf-8', errors='ignore')

    st = time()
    all_ports = [i.strip() for i in args.ports.split(',')]
    parser = parseradd(SERVERS[args.host], all_ports)

    print('Forming zip')
    zipargs = zip(count(0), infile, repeat(parser))
    print('Formed zip')

    with mp.Pool(args.njobs) as pool:
        ans = pool.starmap(parseget, zipargs)
    print('Completed : {} in {} secs'.format(len(ans), time()-st))

    print('Writing results in file {}'.format(args.infile+'-corenlp-opti'))
    for i in ans:
        outfile_corenlp.write(i+'\n')
    print('Written results in file {}'.format(args.infile+'-corenlp-opti'))

    outfile_corenlp.close()
    error_idx.close()
    infile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess dataset to get trees')

    parser.add_argument('-infile', type=str, required=True, help='Specify input file')
    parser.add_argument('-codefile', type = str, required=True, help = 'BPE code file to use')
    parser.add_argument('-word2bpefile', type = str, default = '')
    parser.add_argument('-vocabfile', type = str, default = '')
    parser.add_argument('-host', type=str, required=True, default = 'localhost',
                        help='Which server to check for open ports')
    parser.add_argument('-ports', default='4200,4201,4202,4203,4204,4205,4206', type=str, help='Comma separated port numbers')
    parser.add_argument('-njobs', type=int, default=50, help='Specify number of Parallel jobs')

    args = parser.parse_args()

    codefile = open(args.codefile)

    if args.vocabfile != '':
        with open(args.vocabfile, 'r') as f:
            voc = f.read().split('\n')
            if voc[-1].strip() == '':
                voc = voc[:-1]
            vocab = apply_bpe.read_vocabulary(voc, 0)
    else:
        vocab = None

    bpe_encoder = apply_bpe.BPE(codefile, vocab = vocab)

    if args.word2bpefile != '':
        with open(args.word2bpefile, 'rb') as pk:
            word2bpe = pickle.load(pk)

    else:
        word2bpe = {}

    main(args)
