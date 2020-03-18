import argparse
import os
from src.dataloader import *
from src.helper import *

intent_CLS_FILES = ['intent-classification/yahoo/0', 'intent-classification/yahoo/1', 'intent-classification/trec/1', 'intent-classification/trec/2', 'intent-classification/trec/3','intent-classification/trec/4', 'intent-classification/trec/5', 'intent-classification/trec/0']

def merge_vocabs(voc1, voc2, name):
    voc = Voc(name)
    words = list(set(voc1.w2id.keys()) | set(voc2.w2id.keys()))

    voc.nwords = len(words)
    voc.w2id = {w:i for i,w in enumerate(words)}
    voc.id2w = {i:w for i,w in enumerate(words)}
    voc.w2c = {w: voc1.w2c[w] + voc2.w2c[w] for w in words}

    return voc

def create_vocab_dict(args, voc1, voc2, train_dataloader, test_dataloader=None):
    voc1, voc2 = add_vocab_for(args, voc1, voc2, train_dataloader)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generates vocab.p file')

    parser.add_argument('-dataset', type=str, default='quorafinal', choices=['quorafinal', 'scpn','controlledgen','controlledgen_bpe','quorafinalv2', 'quorabpe'] + intent_CLS_FILES, help='Dataset to use')
    parser.add_argument('-run_name', type=str, default='SYNPar-debug', help='Enter the run name')
    parser.add_argument('-max_length', type=int, default=30, help='Specify max decode steps: Max length string to output')
    parser.add_argument('-single_side', action = 'store_true', help = 'Whether to chain source to target and target to source')
    parser.add_argument('-batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-log_fmt', type=str, default='%(asctime)s | %(levelname)s | %(name)s | %(message)s', help='Specify format of the logger')
    parser.add_argument('-bpe', action = 'store_true', help = 'Use byte pair encoding')
    parser.add_argument('-vocab_size', type=int, default=50000, help='Vocabulary size to consider')

    args = parser.parse_args()

    log_folder_name = os.path.join('Logs', args.run_name)
    create_save_directories('Logs', 'Models',  args.run_name)

    logger = get_logger(__name__, 'temp_run', args.log_fmt, logging.INFO, os.path.join(log_folder_name, 'SYN-Par.log'))
    logger.info('Run name: {}'.format(args.run_name))

    train_set           = IterableParaphraseDataset(args, logger, 'train', True)
    val_set             = IterableValParaphraseDataset(args, logger, 'val')
    train_dataloader    = DataLoader(train_set, batch_size = args.batch_size, num_workers=0, collate_fn = collate)
    val_dataloader      = DataLoader(val_set,   batch_size = args.batch_size, num_workers=0, collate_fn = collateval)

    logger.info('Creating vocab ...')

    voc1 = Voc(args.dataset + 'sents')
    voc2 = Voc(args.dataset + 'trees')
    voc_file = os.path.join('Models', args.run_name, 'vocab.p')

    voc = create_vocab_dict(args, voc1, voc2, train_dataloader)

    logger.info('Vocab created with number of words = {}'.format(voc.nwords))
    logger.info('Saving Vocabulary file')

    with open(voc_file, 'wb') as f:
        pickle.dump(voc, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info('Vocabulary file saved in {}'.format(os.path.join('Models', args.run_name, 'vocab.p')))