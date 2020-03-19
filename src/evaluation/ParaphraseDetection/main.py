import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from src.evaluation.ParaphraseDetection.model import RoBERTaPPD
from src.evaluation.ParaphraseDetection.dataset import ParaphraseDataset, IterableParaphraseDataset
from pytorch_transformers import RobertaTokenizer
from src.evaluation.ParaphraseDetection.args import build_parser
import os
import ipdb as pdb

def find_accuracy(logits, labels):
    probs_np = torch.sigmoid(logits).detach().cpu().numpy().squeeze()
    labels_np = labels.numpy()
    
    probs_np[probs_np > 0.5] = 1
    probs_np[probs_np < 0.5] = 0

    acc = (probs_np == labels_np).mean()
    return acc 

def preprocess_questions(q1_ls, q2_ls, tokenizer):
    '''
    q1_ls = [q1 + ' '.join(['<pad>' for _ in range(max_len_q1 - len(tokenizer.encode(q1)))])\
            for q1 in q1_ls]

    q2_ls = [q2 + ' '.join(['<pad>' for _ in range(max_len_q2 - len(tokenizer.encode(q2)))])\
            for q2 in q2_ls]
    '''
    qpairs = [tokenizer.encode(q1, q2, add_special_tokens = True) for q1,q2 in zip(q1_ls, q2_ls)]
    max_len = max([len(qpair) for qpair in qpairs])
    qpairs_padded = [qpair + [1 for i in range(max_len - len(qpair))] for qpair in qpairs]
    try:
        qpairs_v = torch.LongTensor(qpairs_padded)
    except:
        pdb.set_trace()
    '''
    qpairs_v = torch.cat([torch.LongTensor(tokenizer.encode(q1, q2, add_special_tokens=True)).unsqueeze(0)\
                for q1, q2 in zip(q1_ls, q2_ls)], dim = 0)
    '''
    attn_masks = (qpairs_v != 1).long()
    return qpairs_v, attn_masks

def validate(net, val_loader, criterion, tokenizer, device, logger, ep):

    cnt = 0
    net.eval()
    loss, acc = 0, 0
    with torch.no_grad():
        for q1_ls, q2_ls, labels in val_loader:
            qpairs_v, attn_masks = preprocess_questions(q1_ls, q2_ls, tokenizer)
            if len(qpairs_v) == 0:
                continue
            out = net(qpairs_v.to(device), attn_masks.to(device))
            loss += criterion(out.squeeze(-1), labels.to(device).float()).item()
            acc += find_accuracy(out, labels)
            cnt += 1
    
    print("Epoch {} complete, Validation loss: {}, Validation accuracy: {}"\
                .format(ep, loss / cnt, acc / cnt))
    return acc/cnt

def train(net, train_loader, val_loader, criterion, opti, tokenizer, device, args, logger):
    best_acc = -1
    for ep in range(args.n_epochs):
        net.train()
        running_loss = 0
        running_acc = 0
        for i, (q1_ls, q2_ls, labels) in enumerate(train_loader):
            qpairs_v, attn_masks = preprocess_questions(q1_ls, q2_ls, tokenizer)
            if len(qpairs_v) == 0:
                continue
            opti.zero_grad()
            try:
                out = net(qpairs_v.to(device), attn_masks.to(device))
            except:
                pdb.set_trace()
            loss = criterion(out.squeeze(-1), labels.to(device).float())
            loss.backward()
            opti.step()
            acc = find_accuracy(out, labels)

            if i == 0:
                running_loss = loss.item()
                running_acc = acc
            
            else:
                running_loss = 0.1 * loss.item() + 0.9 * running_loss
                running_acc = 0.1 * acc + 0.9 * running_acc

            if i % args.display_freq == 0:
                print("Epoch {} Batch {} processed, loss = {}, accuracy = {}".\
                            format(ep, i, running_loss, running_acc))

        val_acc = validate(net, val_loader, criterion, tokenizer, device, logger, ep)
        if val_acc > best_acc:
            print("Best accuracy updated from {} to {}".format(best_acc, val_acc))
            best_acc = val_acc
            torch.save(net.state_dict(), 'src/evaluation/ParaphraseDetection/Models/{}_best_{}.dat'.format(args.run_name, np.round(val_acc, 2)))


def test(net, test_loader, tokenizer, device, args, logger):

    cnt = 0
    net.eval()
    mean_acc = 0
    running_acc = 0
    with torch.no_grad():
        for i, (q1_ls, q2_ls, labels) in enumerate(test_loader):
            qpairs_v, attn_masks = preprocess_questions(q1_ls, q2_ls, tokenizer)
            if len(qpairs_v) == 0:
                continue
            out = net(qpairs_v.to(device), attn_masks.to(device))
            acc = find_accuracy(out, labels)
            mean_acc += acc
            if i == 0:
                running_acc = acc
            else:
                running_acc = 0.1 * acc + 0.9 * running_acc
            cnt += 1

            if i % args.display_freq == 0:
                print("Batch {}, accuracy = {}".format(i, running_acc))

    print("Testing complete, Test accuracy: {}"\
                .format(mean_acc / cnt))
    return mean_acc / cnt


if __name__ == '__main__':
    torch.manual_seed(0)
    parser = build_parser()
    args = parser.parse_args()
    device = args.device

    logger = None
    print("Building Model")
    net = RoBERTaPPD().to(device)
    print("Done")

    if args.mode == 'test':
        if 'controlledgen' in args.dataset or 'paranmt' in args.dataset or 'ParaNMT' in args.dataset:
            print("Loading PAWS model")
            net.load_state_dict(torch.load('src/evaluation/ParaphraseDetection/Models/pawsBT_lowlr_best_0.97.dat'))

        else:
            net.load_state_dict(torch.load('src/evaluation/ParaphraseDetection/Models/ppd_best_0.9.dat'))

        #net.load_state_dict(torch.load('src/evaluation/ParaphraseDetection/Models/bt_paws_lowlr_best_0.96.dat'))
        #net.load_state_dict(torch.load('src/evaluation/ParaphraseDetection/Models/PAWS_best_0.95.dat'))
        #net.load_state_dict(torch.load('src/evaluation/ParaphraseDetection/Models_bkp/Models/bt_paws_lowlr_best_0.97.dat'))
        print("Building Test Data loaders")
	    #test_set = IterableParaphraseDataset(args, logger, args.datatype)
        #test_file = os.path.join('src/evaluation/ParaphraseDetection/data',args.dataset,'test_data.csv')
        test_file = args.test_file
        test_set = ParaphraseDataset(test_file)
        test_loader = DataLoader(test_set, batch_size = args.batch_size, num_workers = 0)
        print("Done")
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        print("Let the testing begin")
        test(net, test_loader, tokenizer, device, args, logger)
        print("THE END")        


    else:
        criterion = nn.BCEWithLogitsLoss()
        opti = optim.Adam(net.parameters(), lr = args.lr)


        print("Building Data loaders")
        train_file = os.path.join('src/evaluation/ParaphraseDetection/data',args.dataset,'train_data.csv')
        val_file = os.path.join('src/evaluation/ParaphraseDetection/data',args.dataset,'val_data.csv')    
        train_dataset = ParaphraseDataset(train_file)
        val_dataset = ParaphraseDataset(val_file)
    
        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers = 5, shuffle = True)
        val_loader = DataLoader(val_dataset, batch_size = args.batch_size, num_workers = 5)
        print("Done")
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        print("Let the training begin")
        train(net, train_loader, val_loader, criterion, opti, tokenizer, device, args, logger)
        print("THE END")     
