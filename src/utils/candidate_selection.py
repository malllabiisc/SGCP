import numpy as np
import pandas as pd
import argparse
from src.helper import *
import ipdb as pdb
import time
import os
import nltk
from nltk.tokenize import word_tokenize
import editdistance
import rouge


rouge_eval = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                         max_n=2,
                         limit_length=True,
                         length_limit=100,
                         length_limit_type='words',
                         apply_avg=False,
                         apply_best=False,
                         alpha=0.5,  # Default F1_score
                         weight_factor=1.2,
                         stemming=True)

def select_posed_bleu(src, df_sub):
    poseds = []
    for idx in list(df_sub.index):
        syn = df_sub.loc[idx, 'syn_paraphrase']
        temp = df_sub.loc[idx, 'template']
        syn_tags = list(zip(*nltk.pos_tag(word_tokenize(syn))))[1]
        temp_tags = list(zip(*nltk.pos_tag(word_tokenize(temp))))[1]
        posed = editdistance.eval(syn_tags, temp_tags)
        poseds.append(posed)

    min_posed = min(poseds)
    posed_idx = [i for i in range(len(poseds)) if poseds[i] == min_posed]
    max_bleu = -1
    final_idx = None
    id_start = list(df_sub.index)[0]
    for idx in posed_idx:
        syn = df_sub.loc[id_start + idx, 'syn_paraphrase']
        bleu = bleu_scorer([[src]], [syn])[0]
        if bleu > max_bleu:
            max_bleu = bleu
            final_idx = id_start + idx

    return final_idx

def select_rouge(src, df_sub):
    max_rouge = -1
    max_idx = None
    for idx in list(df_sub.index):
        syn = df_sub.loc[idx, 'syn_paraphrase']
        rouge = rouge_eval.get_scores([syn], [src])['rouge-1'][0]['f'][0]
        if rouge > max_rouge:
            max_rouge = rouge
            max_idx = idx
    return max_idx

def select_bleu(src, df_sub):
    max_bleu = -1
    max_idx = None
    for idx in list(df_sub.index):
        syn = df_sub.loc[idx, 'syn_paraphrase']
        bleu = bleu_scorer([[src]], [syn])[1][0]
        if bleu > max_bleu:
            max_bleu = bleu
            max_idx = idx
    return max_idx

def select_maxht(df_sub):
    max_ht = -1
    max_idx = None
    for idx in list(df_sub.index):
        ht = int(df_sub.loc[idx, 'height'])
        if ht > max_ht:
            max_ht = ht
            max_idx = idx

    return max_idx

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Convert trees file to sentence file')
    parser.add_argument('-mode', default = 'test', help = '')
    parser.add_argument('-gen_dir', required = True, help = ' ')
    parser.add_argument('-clean_gen_file', required = True, help = 'name of the file')
    parser.add_argument('-res_file', required = True, help = 'name of the file')
    parser.add_argument('-crt', choices = ['posed','rouge', 'bleu', 'maxht'], default = 'maxht', help = "Criteria to select best generation")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.gen_dir, args.clean_gen_file))
    srcs_unq = []
    idss = []
    ids = []
    prev_src = None
    prev_temp = None
    it = 0
    for i in range(len(df)):
        if df.loc[i, 'source'] == 'NEXT':
            srcs_unq.append(df.loc[i-1, 'source'])
            idss.append(ids)
            ids = []
        else:
            ids.append(i)

    assert len(idss) == len(srcs_unq)
    elites = []
    for src, ids in zip(srcs_unq, idss):
        df_sub = df.loc[ids]

        if args.crt == 'posed':
            final_idx = select_posed_bleu(src, df_sub)
        elif args.crt == 'bleu':
            final_idx = select_bleu(src, df_sub)
        elif args.crt == 'maxht':
            final_idx = select_maxht(df_sub)
        else:
            final_idx = select_rouge(src, df_sub)
        elites.append(final_idx)

    df_elite = df[df.index.isin(elites)]

    assert len(df_elite) == len(srcs_unq)
    try:
        references = df_elite['reference'].values
    except:
        references = []
    syn_paras = df_elite['syn_paraphrase'].values
    sources = df_elite['source'].values
    #final_bleu = bleu_scorer([list(references)], list(syn_paras))
    #df_elite.to_csv(os.path.join(args.gen_dir, 'elite.csv'))
    with open(os.path.join(args.gen_dir, args.res_file), 'w') as f:
        f.write('\n'.join(syn_paras))







