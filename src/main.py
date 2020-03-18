from __future__ import print_function, division
from __future__ import unicode_literals

from comet_ml import Experiment

import os, math, logging, pdb, sys
from attrdict import AttrDict

import time, json, datetime
import random, argparse
import logging
from tqdm import tqdm
import uuid

from datetime import datetime

from src.args import build_parser
from src.model import SYN_Par
from src.helper import *
from src.dataloader import *

import numpy as np
from collections import OrderedDict

try:
    import cPickle as pickle
except ImportError:
    import pickle

from collections import OrderedDict
import collections

_API_KEY = 'NAN'

def read_files(args, logger):
    if args.mode == 'train':
        logger.info('Training and Validation data loading..')
        if args.len_sort:
            train_set           = IterableParaphraseDataset(args, logger, 'trainsort', True)
        else:
            train_set           = IterableParaphraseDataset(args, logger, 'train', True)

        val_set             = IterableValParaphraseDataset(args, logger, 'val')
        test_set        = IterableValParaphraseDataset(args, logger, 'test')

        train_dataloader    = DataLoader(train_set, batch_size = args.batch_size, num_workers=0, collate_fn = collate)
        val_dataloader      = DataLoader(val_set,   batch_size = args.batch_size, num_workers=0, collate_fn = collateval)
        test_dataloader     = DataLoader(test_set, batch_size = args.batch_size, num_workers=0, collate_fn = collateval)

        logger.info('Training Validation and Test data loaded!')

        return train_dataloader, val_dataloader, test_dataloader

    elif args.mode == 'decode':
        logger.info('Test data loading..')
        test_set            = IterableParaphraseDataset(args, logger, args.datatype)
        test_dataloader     = DataLoader(test_set,  batch_size = args.batch_size, num_workers=0, collate_fn=collate)
        #test_set        = IterableValParaphraseDataset(args, logger, 'test')
        #test_dataloader     = DataLoader(test_set, batch_size = args.batch_size, num_workers=0, collate_fn = collateval)
        logger.info('Test data loaded!')

        return test_dataloader

    else:
        raise Exception('{} is not a valid mode'.format(args.mode))

def prepare_input(args, pairs, voc, device, datatype='train'):
    src_tens, src_extend_tens, oovs = sents2ids(voc, pairs['src'], args.max_length, True)
    src_tens = [pad_seq(seq, args.max_length, voc) for seq in src_tens]
    src_extend_tens = [pad_seq(seq, args.max_length, voc) for seq in src_extend_tens]
    src_tens = torch.LongTensor(src_tens).transpose(0, 1).to(device)
    src_extend_tens = torch.LongTensor(src_extend_tens).transpose(0, 1).to(device)

    para_tens, tgt_tens = paras2ids(voc, pairs['tgt'], args.max_length, oovs)
    tgt_tens = [pad_seq(seq, args.max_length, voc) for seq in tgt_tens]
    para_tens = [pad_seq(seq, args.max_length, voc) for seq in para_tens]
    tgt_tens = torch.LongTensor(tgt_tens).transpose(0, 1).to(device)
    para_tens = torch.LongTensor(para_tens).transpose(0, 1).to(device)

    src = {
        'tens': src_tens,
        'extend_tens': src_extend_tens,
        'oovs': oovs
    }

    tgt = {
        'para_tens': para_tens,
        'tgt_tens': tgt_tens
    }

    return src, tgt

def prune_trees(args, pairs, ht, is_train=True):
    if args.dynamic_tree:
        # PROCESSING FOR DYNAMIC HEIGHT DECODING
        tgt_trees = []
        tgt_phrase_starts = []
        for i in range(len(pairs['tgt_tree'])):
            _, max_tgt_ht = convert(args, json.loads(pairs['tgt_tree'][i]), 'ROOT', 1)
            if is_train:
                if max_tgt_ht < 5:
                    ht = 4
                else:
                    ht = random.choice(range(5, max_tgt_ht+1))
            else:
                ht = max_tgt_ht

            ptreest = trim_trees([pairs['tgt_tree'][i]], ht)
            tgt_tree, tgt_st = ptreest[0][0], ptreest[1][0]
            tgt_trees.append(tgt_tree)
            tgt_phrase_starts.append(tgt_st)
     
    else:
        tgt_trees, tgt_phrase_starts = trim_trees(pairs['tgt_tree'], ht)
    return tgt_trees, tgt_phrase_starts

def validation(args, model, val_dataloader, voc, device, logger):
    refs = []
    hyps = []
    val_loss_epoch = 0.0
    batch_num = 1

    model.eval()
    ht = 6
    logger.info('Running validation')

    for pairs in val_dataloader:
        src, tgt = prepare_input(args, pairs, voc, device, 'validation')

        src_tens, src_extend_tens, oovs = src['tens'], src['extend_tens'], src['oovs']
        tgt_tens = tgt['tgt_tens']
        tgt_trees, _ = prune_trees(args, pairs, ht, False)

        val_loss, decoder_output = model.greedy_decode(tgt_trees,
                                                       src_tens=src_tens,
                                                       src_extend_tens=src_extend_tens,
                                                       oovs=oovs,
                                                       tgt_tens=tgt_tens,
                                                       use_coverage=False,
                                                       is_validation=True)

        # tgt_sents       = ids2sents(voc, oovs, tgt_tens, True)
        hyp_sents       = ids2sents(voc, oovs, decoder_output, True)
        val_loss_epoch += val_loss

        # refs += [' '.join(tgt_sents[i]) for i in range(tgt_tens.size(1))]
        hyps += [bpestr2wordstr(' '.join(hyp_sents[i])) for i in range(tgt_tens.size(1))]

        if args.dataset == 'controlledgen':
            mdf = 5
        else:
            mdf = 50

        # pdb.set_trace()

        # if batch_num % mdf == 0:
        #     logger.info('Sample Generations')
        #     for i in range(5):
        #         strout = '\nSource Sentence: {}\nTemplate Sentence: {}\nSyntactic Paraphrase: {}'
        #         # strout = '\nSource Sentence: {}\nSyntactic Paraphrase: {}'
        #         logger.info(strout.format(' '.join(json.loads(pairs['src'][-i-1])),
        #                                   ' '.join(json.loads(pairs['temp'][-i-1])),
        #                                   hyps[-i-1]))
        batch_num += 1

    return hyps, val_loss_epoch


def one_epoch_train(model, dataloader, voc, device, args, experiment,
                    logger, batch_num, global_step, loss, ep, ep_offset=0):
    train_loss_epoch = 0.0

    ht = random.choice(args.tree_height)
    logger.info('Running for tree_height : {}'.format(ht))

    for i, pairs in enumerate(dataloader):
        if len(pairs['tgt']) == 0:
            continue
        if batch_num % args.display_freq == 0:
            logger.info('Processing Batch : {} , loss : {}'.format(batch_num, loss))

        src, tgt = prepare_input(args, pairs, voc, device, 'train')

        src_tens, src_extend_tens, oovs = src['tens'], src['extend_tens'], src['oovs']
        para_tens, tgt_tens = tgt['para_tens'], tgt['tgt_tens']
        tgt_trees, tgt_phrase_starts = prune_trees(args, pairs, ht)

        loss = model.trainer(para_tens,
                             tgt_tens,
                             tgt_trees,
                             tgt_phrase_starts,
                             src_tens=src_tens,
                             src_extend_tens=src_extend_tens,
                             oovs=oovs,
                             use_coverage = (ep + ep_offset  >= args.cov_after_ep))

        train_loss_epoch += loss
        if not args.debug:
            experiment.log_metric('tp_loss', loss, step=global_step)

        batch_num += 1
        global_step += 1

    return train_loss_epoch, global_step

# Implement training procedure + Validation step
def train(model, train_dataloader, val_dataloader,test_dataloader, voc,
          device, args, logger, ep_offset = 0, min_val_loss=1e8,
          max_val_bleu=0.0, experiment=None):

    val_refs = open('data/{}/val/ref.txt'.format(args.dataset)).read().split('\n')
    if val_refs[-1].strip() == '':
        val_refs = val_refs[:-1]

    val_refs = [bpestr2wordstr(line) for line in val_refs]
    test_refs = open('data/{}/test/ref.txt'.format(args.dataset)).read().split('\n')
    if test_refs[-1].strip() == '':
        test_refs = test_refs[:-1]
    test_refs = [bpestr2wordstr(line) for line in test_refs]

    logger.info('Training Started!!')

    old_path = {'bleu':None, 'val':None, 'bleuval':None}
    start_time = time.time()

    train_loss_epoch = 0
    global_step = 0

    for ep in range(args.max_epochs):
        logger.info('Epoch : {}'.format(ep + ep_offset))

        batch_num = 1
        train_loss_epoch = 0
        loss = 0.0

        model.train()

        train_loss_epoch, global_step = one_epoch_train(model, train_dataloader, voc,
                                                        device, args, experiment,
                                                        logger, batch_num, global_step, loss,
                                                        ep, ep_offset=ep_offset)

        if not args.debug:
            experiment.log_metric('loss/train_loss', train_loss_epoch, step = ep + ep_offset)

        hyps, val_loss_epoch = validation(args, model, val_dataloader, voc, device, logger)

        refs = val_refs
        with open('tempgens/temp_refs.txt-{}'.format(args.run_name), 'w') as f:
            f.write('\n'.join(refs))
        with open('tempgens/temp_hyps.txt-{}'.format(args.run_name), 'w') as f:
            f.write('\n'.join(hyps))
        bleu_score_val = run_multi_bleu('tempgens/temp_hyps.txt-{}'.format(args.run_name), 'tempgens/temp_refs.txt-{}'.format(args.run_name))
        logger.info('Val BLEU scores after epoch {} : {}'.format(ep+ep_offset, bleu_score_val))

        hyps, test_loss_epoch = validation(args, model, test_dataloader, voc, device, logger)
        refs = test_refs
        with open('tempgens/temp_refs.txt-{}'.format(args.run_name), 'w') as f:
            f.write('\n'.join(refs))
        with open('tempgens/temp_hyps.txt-{}'.format(args.run_name), 'w') as f:
            f.write('\n'.join(hyps))
        bleu_score_test = run_multi_bleu('tempgens/temp_hyps.txt-{}'.format(args.run_name), 'tempgens/temp_refs.txt-{}'.format(args.run_name))
        #bleu_score_test = bleu_scorer(refs, hyps)
        #os.remove('temp_hyps.txt-{}'.format(args.run_name))
        #os.remove('temp_refs.txt-{}'.format(args.run_name))
        logger.info('Test BLEU score after epoch {} : {}'.format(ep+ep_offset, bleu_score_test))

        if not args.debug:
            experiment.log_metric('BLEU', bleu_score_test, step=ep+ep_offset)
            experiment.log_metric('loss/validation_loss', test_loss_epoch, step= ep+ep_offset)
            experiment.log_html('epoch:{} \t <br />BLEU:{}<br />'.format(ep + ep_offset,  str(bleu_score_test)))

        if bleu_score_val > max_val_bleu or val_loss_epoch < min_val_loss:
            str_cont = ''
            if bleu_score_val > max_val_bleu:
                max_val_bleu = bleu_score_val
                str_cont += 'bleu'
            if val_loss_epoch < min_val_loss:
                min_val_loss = val_loss_epoch
                str_cont += 'val'
            state = {
                'epoch' : ep + ep_offset,
                'model_state_dict': model.state_dict(),
                'voc': model.voc,
                'optimizer_rnn_state_dict': model.optimizer['rnn'].state_dict(),
                'optimizer_tree_state_dict': model.optimizer['tree'].state_dict(),
                'train_loss' : train_loss_epoch,
                'val_loss' : val_loss_epoch,
                'bleu' : bleu_score_val
            }
            if args.pretrained_encoder is None:
                state['optimizer_enc_state_dict'] = model.optimizer['encoder'].state_dict()

            # SAVE CHECKPOINT
            save_checkpoint(state, ep+ep_offset,
                            logger, os.path.join('Models',
                                                 args.run_name,
                                                 '{}-{}'.format(str_cont, args.ckpt_file)),
                            args, old_path[str_cont])

            ckpt_name = '{}_{}.pth.tar'.format('{}-{}'.format(str_cont, args.ckpt_file).split(".")[0].split("_")[0], ep + ep_offset)
            old_path[str_cont] = os.path.join('Models', args.run_name, ckpt_name)

        # Validation code after each epoch
        logger.info('Train Loss after epoch {} : {}'.format(ep + ep_offset, train_loss_epoch))
        logger.info('Validation loss after {} epoch: {}'.format(ep + ep_offset, val_loss_epoch))

    logger.info('Training Completed!')

# Greedy Decoding
def decode_greedy(model, test_dataloader,  voc, device, args, logger):
    logger.info('Test Generations')
    result_dir = os.path.join(args.res_folder, args.run_name)
    results_file = os.path.join(result_dir, args.res_file)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    model.eval()
    ht = args.height_dec

    with open(results_file, 'w', encoding='utf-8', errors='ignore') as f:
        refs1 = []
        refs2 = []
        hyps = []
        for pairs in test_dataloader:
            src, tgt = prepare_input(args, pairs, voc, device, 'test')
            src_tens, src_extend_tens, oovs = src['tens'], src['extend_tens'], src['oovs']
            tgt_tens = tgt['tgt_tens']

            src_sents_all = []
            src_tens_all = []
            src_extend_tens_all = []
            oovs_all = []
            tgt_tens_all = []
            tgt_trees_all = []
            hts = []
            tgt_lens = []
            num_hts = []
            num_hts = []
            for i in range(len(pairs['tgt_tree'])):
                _, tgt_ht = convert(args, json.loads(pairs['tgt_tree'][i]), 'ROOT', 1)
                htmin = max(3, tgt_ht - 4)
                htmax = max(3, tgt_ht)
                #htmin = htmax
                #htmax = 6
                num_hts.append(htmax - htmin + 1)
                for ht in range(htmin, htmax + 1):
                    tgt_tree = trim_trees([pairs['tgt_tree'][i]], ht)[0][0]
                    tgt_trees_all.append(tgt_tree)
                    src_sents_all.append(pairs['src'][i])
                    src_tens_all.append(src_tens[:,i:i+1])
                    src_extend_tens_all.append(src_extend_tens[:,i:i+1])
                    oovs_all.append(oovs[i])
                    tgt_tens_all.append(tgt_tens[:,i:i+1])
                    hts.append(ht)

            src_tens_all = torch.cat(src_tens_all, dim = 1)
            src_extend_tens_all = torch.cat(src_extend_tens_all, dim = 1)
            tgt_tens_all = torch.cat(tgt_tens_all, dim = 1)
            cum_num_hts = [sum(num_hts[:i+1]) for i in range(len(num_hts))]
            #tgt_trees, tgt_phrase_starts = trim_trees(pairs['tgt_tree'], ht)
            if args.beam_width == 1:
                val_loss, decoder_output = model.greedy_decode(tgt_trees_all,
                                                            src_tens=src_tens_all,
                                                            src_extend_tens=src_extend_tens_all,
                                                            oovs=oovs_all,
                                                            tgt_tens=None,
                                                            use_coverage=False,
                                                            is_validation=False)
                tgt_sents       = ids2sents(voc, oovs_all, tgt_tens_all, True)
                hyp_sents       = ids2sents(voc, oovs_all, decoder_output, True)

            else:
                val_loss, decoder_output = model.beam_decode_naive(tgt_trees_all,
                                                            src_tens=src_tens_all,
                                                            src_extend_tens=src_extend_tens_all,
                                                            oovs=oovs_all,
                                                            tgt_tens=None,
                                                            use_coverage=False,
                                                            is_validation=False)

                tgt_sents       = ids2sents(voc, oovs_all, tgt_tens_all, True)
                hyp_sents       = lsids2sents(voc, oovs_all, decoder_output, True)

            it = 0
            for i in range(len(tgt_sents)):
                if i == cum_num_hts[it]:
                    it += 1
                    f.write('********************\n')
                    logger.info("********************")

                f.write('Height: {} \n'.format(hts[i]))
                f.write('Source Sentence: {} \n'.format(bpestr2wordstr(' '.join(json.loads(src_sents_all[i])))))
                f.write('Template Sentence: {} '.format(bpestr2wordstr(' '.join(tgt_sents[i]))))
                f.write('Syntactic Paraphrase : {}\n'.format(bpestr2wordstr(' '.join(hyp_sents[i]))))
                f.write('-----------------\n')

                logger.info('Height: {} \n'.format(hts[i]))
                logger.info('Source Sentence: {}'.format(bpestr2wordstr(' '.join(json.loads(src_sents_all[i])))))
                logger.info('Template Sentence: {} '.format(bpestr2wordstr(' '.join(tgt_sents[i]))))
                logger.info('Syntactic Paraphrase: {}'.format(bpestr2wordstr(' '.join(hyp_sents[i]))))
                logger.info('-------------------------')

            f.write('********************\n')
            logger.info("********************")

            refs1 += [[' '.join(tgt_sents[i])] for i in range(tgt_tens_all.size(1))]
            hyps += [' '.join(hyp_sents[i]) for i in range(tgt_tens_all.size(1))]
            refs2 += [[' '.join(json.loads(src_sents_all[i]))] for i in range(tgt_tens_all.size(1))]

        bleu_scores = bleu_scorer(refs2, hyps)
        logger.info('Bleu score with source sentences: {}'.format(bleu_scores[0]))
        bleu_scores = bleu_scorer(refs1, hyps)
        logger.info('Bleu score with template sentences: {}'.format(bleu_scores[0]))

    logger.info('Decoding Complete!!')


def merge_vocabs(voc1, voc2, name):
    voc = Voc(name)
    words = list(set(voc1.w2id.keys()) | set(voc2.w2id.keys()))
    for word in words:
        voc.addWord(word)
    
    return voc

def create_vocab_dict(args, voc1, voc2, train_dataloader, test_dataloader=None):
    voc1, voc2 = add_vocab_for(args, voc1, voc2, train_dataloader)
    # voc1, voc2 = add_vocab_for(args, voc1, voc2, val_dataloader)
    if test_dataloader != None:
        voc1, voc2 = add_vocab_for(args, voc1, voc2, test_dataloader)
    voc1.most_frequent(args.vocab_size)
    voc = merge_vocabs(voc1, voc2, args.dataset)
    assert len(voc.w2id) == voc.nwords
    assert len(voc.id2w) == voc.nwords
    return voc

def add_vocab_for(args, voc1, voc2, dataloader):
    for pairs in dataloader:
        for src, src_tree in zip(pairs['src'], pairs['src_tree']):
            src_tree = json.loads(src_tree)
            src = json.loads(src)
            voc1.addSentence(src)
            voc2.addTree(src_tree, 'ROOT')
    return voc1, voc2

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.tree_height = [int(s) for s in args.tree_height.split(",")]
    use_ptr = args.use_ptr
    cov_after_ep = args.cov_after_ep
    height_dec = args.height_dec


    if args.mode == 'train':
        if len(args.run_name.split()) == 0:
            args.run_name = datetime.fromtimestamp(time.time()).strftime(args.date_fmt)

    if args.pretrained_encoder == None or args.pretrained_encoder == 'bert_all':
        args.use_attn = True

    # SET SEEDS
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ASSIGN GPU
    device = gpu_init_pytorch(args.gpu)
    # CREATE LOGGING FOLDER
    log_folder_name = os.path.join('Logs', args.run_name)
    create_save_directories('Logs', 'Models',  args.run_name)

    # Comet ML - Log all params
    experiment = None
    if not args.debug:
        experiment = Experiment(api_key=_API_KEY, project_name=args.project_name, workspace="NAN")
        experiment.set_name(args.run_name)
        experiment.log_parameters(vars(args))

    logger = get_logger(__name__, 'temp_run', args.log_fmt, logging.INFO, os.path.join(log_folder_name, 'SYN-Par.log'))
    logger.info('Run name: {}'.format(args.run_name))

    if args.mode == 'train':
        train_dataloader, val_dataloader, test_dataloader = read_files(args, logger)
        logger.info('Creating vocab ...')

        voc1 = Voc(args.dataset + 'sents')
        voc2 = Voc(args.dataset + 'trees')
        voc_file = os.path.join('Models', args.run_name, 'vocab.p')

        if (os.path.exists(voc_file)):
            logger.info('Loading vocabulary from {}'.format(os.path.join('Models', args.run_name, 'vocab.p')))
            voc = pickle.load(open(voc_file, 'rb'))
        else:
            voc = create_vocab_dict(args, voc1, voc2, train_dataloader)

            logger.info('Vocab created with number of words = {}'.format(voc.nwords))
            logger.info('Saving Vocabulary file')

            with open(voc_file, 'wb') as f:
                pickle.dump(voc, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info('Vocabulary file saved in {}'.format(os.path.join('Models', args.run_name, 'vocab.p')))
    else:
        config_file_name = os.path.join('Models', args.run_name, 'config.p')
        mode = args.mode
        batch_size = args.batch_size
        beam_width = args.beam_width
        gpu = args.gpu
        tree_height2 = 40
        use_glove = args.use_glove
        max_length = args.max_length
        datatype = args.datatype
        res_file = args.res_file
        dataset = args.dataset
        load_from_ep = args.load_from_ep
        run_name = args.run_name
        max_epochs = args.max_epochs 

        with open(config_file_name, 'rb') as f:
            args = AttrDict(pickle.load(f))

        args.mode = mode
        args.gpu = gpu
        args.load_from_ep = load_from_ep
        args.dataset = dataset
        args.beam_width = beam_width
        args.gpu = gpu
        args.height_dec = height_dec
        args.tree_height2 = tree_height2
        args.use_glove = use_glove
        args.max_length = max_length
        args.datatype = datatype
        args.res_file = res_file
        args.run_name = run_name
        args.max_epochs = max_epochs


        test_dataloader = read_files(args, logger)

        logger.info('Loading Vocabulary file')
        with open(os.path.join('Models', args.run_name, 'vocab.p'), 'rb') as f:
            voc = pickle.load(f)
        logger.info('Vocabulary file Loaded from {}'.format(os.path.join('Models', args.run_name, 'vocab.p')))

    checkpoint = get_latest_checkpoint('Models', args.run_name, logger, args.load_from_ep)

    if args.mode == 'train':
        if checkpoint == None:
            logger.info('Starting a fresh training procedure')
            ep_offset = 0
            min_val_loss = 1e8
            max_val_bleu = 0.0
            config_file_name = os.path.join('Models', args.run_name, 'config.p')
            if args.use_word2vec:
                logger.info('Over-writing emb_size to 300 because argument use_word2vec has been set to True')
                args.emb_size = 300

            model = SYN_Par(args, voc, device, logger, experiment)
            with open(config_file_name, 'wb') as f:
                pickle.dump(vars(args), f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            config_file_name = os.path.join('Models', args.run_name, 'config.p')
            debug = args.debug
            max_epochs = args.max_epochs
            with open(config_file_name, 'rb') as f:
                args = AttrDict(pickle.load(f))

            if args.use_word2vec:
                logger.info('Over-writing emb_size to 300 because argument use_word2vec has been set to True')
                args.emb_size = 300
            args.use_ptr = use_ptr
            args.debug = debug
            args.cov_after_ep = cov_after_ep
            args.max_epochs = max_epochs

            model = SYN_Par(args, voc, device, logger, experiment)
            ep_offset, train_loss, min_val_loss, max_val_bleu, voc = load_checkpoint(model,
                                                                                     args.mode,
                                                                                     checkpoint,
                                                                                     logger,
                                                                                     device,
                                                                                     args.pretrained_encoder)

            logger.info('Resuming Training From ')
            od = OrderedDict()
            if ep_offset is None and train_loss is None and min_val_loss is None and max_val_bleu is None:
                od['Epoch'] = 0
                od['Train_loss'] = 0.0
                od['Validation_loss'] = 0.0
                od['Validation_Bleu'] = 0.0
            else:
                od['Epoch'] = ep_offset
                od['Train_loss'] = train_loss
                od['Validation_loss'] = min_val_loss
                od['Validation_Bleu'] = float(max_val_bleu)
            print_log(logger, od)
            ep_offset += 1
            max_val_bleu = float(max_val_bleu)

        train(model, train_dataloader, val_dataloader, test_dataloader,
              voc, device, args, logger, ep_offset,
              min_val_loss, max_val_bleu, experiment=experiment)
    else:
        if checkpoint == None:
            logger.info('Cannot decode because of absence of checkpoints')
            sys.exit()
        else:
            config_file_name = os.path.join('Models', args.run_name, 'config.p')
            beam_width = args.beam_width
            gpu = args.gpu
            tree_height2 = 40
            use_glove = args.use_glove
            max_length = args.max_length
            datatype = args.datatype
            res_file = args.res_file
            dataset = args.dataset
            load_from_ep = args.load_from_ep
            run_name = args.run_name
            max_epochs = args.max_epochs
            with open(config_file_name, 'rb') as f:
                args = AttrDict(pickle.load(f))

            args.load_from_ep = load_from_ep
            args.dataset = dataset
            args.beam_width = beam_width
            args.gpu = gpu
            args.height_dec = height_dec
            args.tree_height2 = tree_height2
            args.use_glove = use_glove
            args.max_length = max_length
            args.datatype = datatype
            args.res_file = res_file
            args.run_name = run_name
            args.max_epochs = max_epochs
            if args.use_word2vec:
                logger.info('Over-writing emb_size to 300 because argument use_word2vec has been set to True')
                args.emb_size = 300

            model = SYN_Par(args, voc, device, logger, experiment)
            ep_offset, train_loss, min_val_loss, max_val_bleu, voc = load_checkpoint(model,
                                                                                     args.mode,
                                                                                     checkpoint,
                                                                                     logger,
                                                                                     device,
                                                                                     args.pretrained_encoder)
            logger.info('Decoding from')
            od = OrderedDict()

            if ep_offset is None and train_loss is None and min_val_loss is None and max_val_bleu is None:
                od['Epoch'] = 0
                od['Train_loss'] = 0.0
                od['Validation_loss'] = 0.0
                od['Validation_Bleu'] = 0.0
            else:
                od['Epoch'] = ep_offset
                od['Train_loss'] = train_loss
                od['Validation_loss'] = min_val_loss
                od['Validation_Bleu'] = float(max_val_bleu)
            print_log(logger, od)

        '''
        refs, hyps, test_loss_epoch = validation(args, model, test_dataloader, voc, device, logger)
        refs = open('data/controlledgen/test_ref.txt').read().split('\n')[:-1]
        with open('temp_refs.txt', 'w') as f:
            f.write('\n'.join(refs))
        with open('temp_hyps.txt', 'w') as f:
            f.write('\n'.join(hyps))
        bleu_score_test = run_multi_bleu('temp_hyps.txt', 'temp_refs.txt')
        #bleu_score_test = bleu_scorer(refs, hyps)
        #os.remove('temp_hyps.txt')
        #os.remove('temp_refs.txt')

        logger.info('Test BLEU score: {}'.format(bleu_score_test))
        '''
        decode_greedy(model, test_dataloader, voc, device, args, logger)

if __name__ == "__main__":
    main()
