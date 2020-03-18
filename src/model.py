import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torch_geometric.nn import GCNConv
# from torch_sparse import spmm, transpose
# from scipy.linalg import block_diag
# from scipy.sparse import csr_matrix
import random

import sys
#import tensorflow as tf
#import tensorflow_hub as hub

from src.models.encoderrnn import *
from src.models.treeencoder import *
from src.models.pointergen import *
from src.models.attention import *
from src.helper import *

import ipdb as pdb
from heapq import *
import copy

class BeamNode:
    def __init__(self, decoder_input, decoder_hidden,leaf_embedding_list, decoded_words):
        self.decoder_input = decoder_input
        self.decoded_words = decoded_words
        self.decoder_hidden = decoder_hidden
        self.leaf_embedding_list = leaf_embedding_list

    def __gt__(self, other):
        if len(self.decoded_words) >= len(other.decoded_words):
            return True
        elif len(self.decoded_words) < len(other.decoded_words):
            return False

    def __lt__(self, other):
        if len(self.decoded_words) < len(other.decoded_words):
            return True
        elif len(self.decoded_words) >= len(other.decoded_words):
            return False

class SYN_Par(nn.Module):
    def __init__(self, config, voc, device, logger, EOS_tag='EOS', SOS_tag='SOS'):
        super(SYN_Par, self).__init__()

        self.config         = config
        self.device         = device
        self.voc            = voc
        self.EOS_tag        = EOS_tag
        self.SOS_tag        = SOS_tag
        self.SOS_token      = self.voc.w2id['SOS']
        self.EOS_token      = self.voc.w2id['EOS']
        self.logger         = logger


        self.embedding  = nn.Embedding(self.voc.nwords, self.config.emb_size)
        # INITIALIZING WORD EMBEDDING MATRIX
        if self.config.use_word2vec:
            try:
                logger.info('Initializing word vectors from pretrained model: word2vec....')
                initialize_embeddings(self.embedding, os.path.join('data', 'wordvecs', 'word2vec'), 'GoogleNews-vectors-negative300.bin', self.voc, self.device, 'word2vec')
                logger.info('Word Vectors initialized')
            except:
                logger.info('Please place the word2vec binary file in data/wordvecs/word2vec/GoogleNews-vectors-negative300.bin')

        elif self.config.use_glove:
            logger.info('Initializing word vectors from pretrained model: glove....')
            initialize_embeddings(self.embedding, os.path.join('data', 'wordvecs', 'glove'), 'glove.6B.{}d.txt-pkl'.format(self.config.emb_size), self.voc, self.device, 'glove')
            logger.info('Word Vectors initialized')

        else:
            # Based on AWD-LSTM
            nn.init.uniform_(self.embedding.weight, -1*self.config.init_range, self.config.init_range)

        # Fill params for encoder and decoder
        self.logger.info('Building Encoder RNN..')
        self.encoder        = EncoderRNN(self.config.hidden_size,
                                         self.embedding,
                                         self.config.cell_type,
                                         self.config.depth,
                                         self.config.s2sdprate,
                                         self.config.bidirectional).to(device)

        self.logger.info('Encoder RNN built')

        self.logger.info('Building Decoder...')

        self.treeencoder = TreeEncoder(self.config,
                                       self.embedding,
                                       self.voc,
                                       self.device).to(device)

        self.logger.info('Decoder RNN built')

        self.optimizer = {}
        self.params = {}
        self._optim()

        # Specify criterion
        self.criterion = nn.NLLLoss()
        self.criterion_eop = nn.BCEWithLogitsLoss()

    def _optim(self):
        self._optimer(self.treeencoder.transform, 'tree', self.config.lr_tree)
        self._optimer(self.treeencoder.leaf_forward, 'rnn', self.config.lr_rnndec)
        if self.config.pretrained_encoder is None:
            self._optimer(self.encoder, 'encoder', self.config.lr_enc)

    def _optimer(self, net, name, lr=0.0002):
        self.params[name] = list(net.parameters())
        if self.config.opt     == 'adam':
            self.optimizer[name] = torch.optim.Adam(self.params[name], lr=lr)
        elif self.config.opt   == 'adadelta':
            self.optimizer[name] = torch.optim.Adadelta(self.params[name], lr=lr)
        elif self.config.opt   == 'asgd':
            self.optimizer[name] = torch.optim.ASGD(self.params[name], lr=lr)
        else:
            self.optimizer[name] = torch.optim.SGD(self.params[name], lr=lr)

    def trainer(self, para_tens, tgt_tens, tgt_trees, tgt_phrase_starts,
                src_tens = None, src_extend_tens=None, oovs=None,
                src_embed = None, use_coverage = False):

        #target_tens is still max_sentence_len x batch_len
        for k in self.optimizer:
            self.optimizer[k].zero_grad()

        attn_inputs = None

        src_lens = torch.sum(src_tens != self.voc.w2id['EOS'], dim=0)
        encoder_output, encoder_hidden = self.encoder(src_tens, src_lens)
        encoder_outputs = torch.max(encoder_output, dim = 0)[0]
        attn_masks = (src_tens == self.voc.w2id['EOS']).transpose(0, 1)[:, :encoder_output.size(0)]
        attn_inputs = encoder_output

        leaf_embedding_list = self.treeencoder(encoder_outputs, tgt_trees)

        batch_size = encoder_outputs.size()[0]

        decoder_input_1 = torch.stack([lst[0] for lst in leaf_embedding_list]) # batch_len x d
        decoder_input_2 = para_tens[0,:] # batch_len x 1
        decoder_input_2 = self.embedding(decoder_input_2) # batch_len x d
        decoder_input = torch.cat((decoder_input_1, decoder_input_2), 1) # batch_len x 2d
        decoder_hidden = encoder_outputs.unsqueeze(0).repeat(self.config.depth, 1, 1)

        phrase_markers = [0 for _ in range(batch_size)] # for inputting appropriate parent leaf node at every step in decoder

        self.loss = 0
        cov_vec = torch.zeros(attn_inputs.size(0), batch_size).type(torch.FloatTensor).to(self.device)
        enc_batch_extend_vocab = torch.zeros((encoder_output.size()[0], batch_size)).transpose(0,1).to(self.device)

        src_extend_tens = src_extend_tens.transpose(0,1)

        for i, enc in enumerate(src_extend_tens):
            enc_batch_extend_vocab[i, :src_lens[i]] = src_extend_tens[i, :src_lens[i]]

        enc_batch_extend_vocab = enc_batch_extend_vocab.transpose(0,1)
        enc_batch_extend_vocab = Variable(enc_batch_extend_vocab.long())
        enc_batch_extend_vocab = enc_batch_extend_vocab.to(self.device)

        extra_zeros = None
        if oovs is not None:
            max_oovs = max(len(oov) for oov in oovs)
            extra_zeros = Variable(torch.zeros((batch_size, max_oovs))).to(self.device)

        true_tgt_labels = torch.zeros((src_tens.size()[0], batch_size)).transpose(0,1).to(self.device)
        for i, phrase_start_list in enumerate(tgt_phrase_starts):
            true_tgt_labels[i, phrase_start_list] = 1
            # if src_lens[i].item() < src_tens.size()[0]:
                # true_tgt_labels[i, src_lens[i].item()] = 1
            true_tgt_labels[i] = torch.cat((true_tgt_labels[i][1:], torch.zeros((1)).to(self.device)),0)
        true_tgt_labels = true_tgt_labels.transpose(0,1)

        # print(true_tgt_labels.shape)
        # pdb.set_trace()

        use_teacher_forcing = random.random() < self.config.tfr

        if not use_coverage:
            cov_vec = None

        for di in range(self.config.max_length):
            # CURRENTLY WILL WORK WITH ONLY -USE_PTR
            #
            decoder_output, decoder_hidden, decoder_attention, sig_hid = self.treeencoder.leaf_forward(decoder_input,
                                                                                                       decoder_hidden,
                                                                                                       attn_inputs,
                                                                                                       enc_batch_extend_vocab,
                                                                                                       extra_zeros,
                                                                                                       attn_masks, None)
            if use_coverage:
                self.loss += self.config.cov_wt * torch.sum(torch.min(cov_vec, decoder_attention))
                cov_vec = cov_vec + decoder_attention

            decoder_input_1 = []
            # for i, phrase_start_list in enumerate(tgt_phrase_starts):
            #     if phrase_markers[i] + 1 < len(phrase_start_list) and di == phrase_start_list[phrase_markers[i] + 1]:
            #         phrase_markers[i] += 1
            #     decoder_input_1.append(leaf_embedding_list[i][phrase_markers[i]])
            #
            for i, phrase_start_list in enumerate(tgt_phrase_starts):
                if true_tgt_labels[di, i].item() == 1:
                    if len(leaf_embedding_list[i]) > 1:
                        leaf_embedding_list[i] = leaf_embedding_list[i][1:]
                decoder_input_1.append(leaf_embedding_list[i][0])


            decoder_input_1 = torch.stack(decoder_input_1)
            topv, topi = decoder_output.topk(1)

            self.loss += self.criterion(decoder_output, tgt_tens[di])
            self.loss += self.config.lambda_dec_eop*10.0*self.criterion_eop(sig_hid.squeeze(-1), true_tgt_labels[di])

            # TEACHER FORCING
            if use_teacher_forcing and di < (self.config.max_length - 1):
                try:
                    decoder_input_2 = self.embedding(para_tens[di+1, :])
                except:
                    pdb.set_trace()
            else:
                decoder_input_2 = topi.squeeze(1).detach()
                get_idx = (decoder_input_2 >= self.voc.nwords).nonzero()
                decoder_input_2[get_idx] = self.voc.w2id['UNK']
                try:
                    decoder_input_2 = self.embedding(decoder_input_2)
                except:
                    pdb.set_trace()

            decoder_input = torch.cat((decoder_input_1, decoder_input_2), 1) # batch_len x 2d

        self.loss.backward()

        if self.config.max_grad_norm > 0:
            for k in self.params:
                torch.nn.utils.clip_grad_norm_(self.params[k], self.config.max_grad_norm)

        for k in self.optimizer:
            self.optimizer[k].step()

        return self.loss.item()/self.config.max_length

    def beam_decode_naive(self, tgt_trees, src_tens = None, src_extend_tens=None, oovs=None, tgt_tens=None, use_coverage=None, is_validation=False):
        with torch.no_grad():
            attn_inputs = None
            src_lens = torch.sum(src_tens != self.voc.w2id['EOS'], dim=0)
            encoder_output, encoder_hidden = self.encoder(src_tens, src_lens)
            encoder_outputs = torch.max(encoder_output, dim = 0)[0]
            attn_masks = (src_tens == self.voc.w2id['EOS']).transpose(0, 1)[:, :encoder_output.size(0)]
            attn_inputs = encoder_output

            leaf_embedding_list = self.treeencoder(encoder_outputs, tgt_trees)
            batch_size = encoder_outputs.size()[0]

            decoder_input_1 = torch.stack([lst[0] for lst in leaf_embedding_list]) # batch_len x d
            decoder_input_2 = torch.LongTensor([self.SOS_token for i in range(batch_size)]).to(self.device) # batch_len x 1
            decoder_input_2 = self.embedding(decoder_input_2) # batch_len x d
            decoder_input = torch.cat((decoder_input_1, decoder_input_2), 1) # batch_len x 2d
            decoder_hidden = encoder_outputs.unsqueeze(0).repeat(self.config.depth, 1, 1)

            phrase_markers = [0 for _ in range(batch_size)] # for inputting appropriate parent leaf node at every step in decoder

            self.loss = 0
            cov_vec = torch.zeros(attn_inputs.size(0), batch_size).type(torch.FloatTensor).to(self.device)
            enc_batch_extend_vocab = torch.zeros((encoder_output.size()[0], batch_size)).transpose(0,1).to(self.device)

            src_extend_tens = src_extend_tens.transpose(0,1)
            for i, enc in enumerate(src_extend_tens):
                enc_batch_extend_vocab[i, :src_lens[i]] = src_extend_tens[i, :src_lens[i]]

            enc_batch_extend_vocab = enc_batch_extend_vocab.transpose(0,1)
            enc_batch_extend_vocab = Variable(enc_batch_extend_vocab.long())
            enc_batch_extend_vocab = enc_batch_extend_vocab.to(self.device)

            extra_zeros = None
            if oovs is not None:
                max_oovs = max(len(oov) for oov in oovs)
                extra_zeros = Variable(torch.zeros((batch_size, max_oovs))).to(self.device)


            ret_words = []
            for batch in range(batch_size):
                decoder_input_b = decoder_input[batch:batch+1] #[2d, ]
                decoder_hidden_b = decoder_hidden[:,batch:batch+1] #[2d_h, ]
                leaf_embedding_list_b = leaf_embedding_list[batch:batch+1]
                cov_vec_b = cov_vec[:, batch: batch+1]

                pq = []
                decoded_beams = []
                node = BeamNode(decoder_input_b, decoder_hidden_b, leaf_embedding_list_b, [self.SOS_token])
                heappush(pq, (0, node))
                done = False
                while len(pq) != 0:
                    if len(decoded_beams) == 2 * self.config.beam_width:
                        break
                    scores_n_nodes = [heappop(pq) for _ in range(self.config.beam_width) if len(pq) != 0]
                    #Clear the priority queue after selecting the best k nodes
                    pq = []

                    for score_n_node in scores_n_nodes:
                        score, node = score_n_node

                        if node.decoded_words[-1] == self.EOS_token or len(node.decoded_words) == self.config.max_length:
                            decoded_beams.append((score, node))
                            continue

                        decoder_output, decoder_hidden_b, decoder_attention, sig_hid = self.treeencoder.leaf_forward(
                            node.decoder_input,
                            node.decoder_hidden.contiguous(),
                            attn_inputs[:,batch:batch+1],
                            enc_batch_extend_vocab[:,batch:batch+1],
                            extra_zeros[batch:batch+1, :],
                            attn_masks[batch:batch+1, :],
                            None)

                        decoder_input_1 = []

                        for i in range(len(node.leaf_embedding_list)):
                            if F.sigmoid(sig_hid[i]).item() > 0.5:
                                if len(node.leaf_embedding_list[i]) > 1:
                                    node.leaf_embedding_list[i] = node.leaf_embedding_list[i][1:]
                            decoder_input_1.append(node.leaf_embedding_list[i][0])
                        decoder_input_1 = torch.stack(decoder_input_1)
                        topv, topi = decoder_output.topk(self.config.beam_width)
                        '''
                        if len(node.decoded_words) - 1 < (self.config.max_length - 1):
                            self.loss += self.criterion(decoder_output, tgt_tens[len(node.decoded_words) - 1])
                        '''
                        #topv, topi = topv.squeeze(), topi.squeeze()

                        for i in range(topv.shape[1]):

                            decoder_input_2 = topi[:,i].detach()
                            word_token = decoder_input_2[0].clone().item()
                            get_idx = (decoder_input_2 >= self.voc.nwords).nonzero()
                            decoder_input_2[get_idx] = self.voc.w2id['UNK']

                            decoder_input_embed = self.embedding(decoder_input_2)
                            decoder_input_b = torch.cat((decoder_input_1, decoder_input_embed), 1) # batch_len x 2d
                            new_node = BeamNode(decoder_input_b, decoder_hidden_b,copy.deepcopy(node.leaf_embedding_list), node.decoded_words + [word_token])
                            new_score = -score + topv[0, i].item()
                            try:
                                heappush(pq, (-new_score, new_node))
                            except:
                                pdb.set_trace()
                batch_out = min([beam for beam in decoded_beams if beam[0] != 0], key = lambda x : x[0] / len(x[1].decoded_words))[1].decoded_words
                ret_words.append(batch_out[1:])
                if len(batch_out[1:]) == 0:
                    pdb.set_trace()
                torch.cuda.empty_cache()

        return 0, ret_words

    def greedy_decode(self, tgt_trees, src_tens = None,
                      src_extend_tens=None, oovs=None,
                      tgt_tens=None, use_coverage=None, is_validation=False):
        for k in self.optimizer:
            self.optimizer[k].zero_grad()

        with torch.no_grad():
            attn_inputs = None
            src_lens = torch.sum(src_tens != self.voc.w2id['EOS'], dim=0)
            encoder_output, encoder_hidden = self.encoder(src_tens, src_lens)
            encoder_outputs = torch.max(encoder_output, dim = 0)[0]
            attn_masks = (src_tens == self.voc.w2id['EOS']).transpose(0, 1)[:, :encoder_output.size(0)]
            attn_inputs = encoder_output

            leaf_embedding_list = self.treeencoder(encoder_outputs, tgt_trees)
            batch_size = encoder_outputs.size()[0]

            decoder_input_1 = torch.stack([lst[0] for lst in leaf_embedding_list]) # batch_len x d
            decoder_input_2 = torch.LongTensor([self.SOS_token for i in range(batch_size)]).to(self.device) # batch_len x 1
            decoder_input_2 = self.embedding(decoder_input_2) # batch_len x d
            decoder_input = torch.cat((decoder_input_1, decoder_input_2), 1) # batch_len x 2d
            decoder_hidden = encoder_outputs.unsqueeze(0).repeat(self.config.depth, 1, 1)

            phrase_markers = [0 for _ in range(batch_size)] # for inputting appropriate parent leaf node at every step in decoder

            self.loss = 0
            cov_vec = torch.zeros(attn_inputs.size(0), batch_size).type(torch.FloatTensor).to(self.device)
            enc_batch_extend_vocab = torch.zeros((encoder_output.size()[0], batch_size)).transpose(0,1).to(self.device)

            src_extend_tens = src_extend_tens.transpose(0,1)
            for i, enc in enumerate(src_extend_tens):
                enc_batch_extend_vocab[i, :src_lens[i]] = src_extend_tens[i, :src_lens[i]]

            enc_batch_extend_vocab = enc_batch_extend_vocab.transpose(0,1)
            enc_batch_extend_vocab = Variable(enc_batch_extend_vocab.long())
            enc_batch_extend_vocab = enc_batch_extend_vocab.to(self.device)

            extra_zeros = None
            if oovs is not None:
                max_oovs = max(len(oov) for oov in oovs)
                extra_zeros = Variable(torch.zeros((batch_size, max_oovs))).to(self.device)


            ret_words = []
            for di in range(self.config.max_length):
                decoder_output, decoder_hidden, decoder_attention, sig_hid = self.treeencoder.leaf_forward(decoder_input,
                                                                                                           decoder_hidden,
                                                                                                           attn_inputs,
                                                                                                           enc_batch_extend_vocab,
                                                                                                           extra_zeros,
                                                                                                           attn_masks,
                                                                                                           None)
                decoder_input_1 = []

                for i in range(len(leaf_embedding_list)):
                    if F.sigmoid(sig_hid[i]).item() > 0.5:
                        if len(leaf_embedding_list[i]) > 1:
                            leaf_embedding_list[i] = leaf_embedding_list[i][1:]
                    decoder_input_1.append(leaf_embedding_list[i][0])
                decoder_input_1 = torch.stack(decoder_input_1)

                topv, topi = decoder_output.topk(1)

                if di < (self.config.max_length - 1) and is_validation:
                    self.loss += self.criterion(decoder_output, tgt_tens[di])
                decoder_input_2 = topi.squeeze(1).detach()
                ret_words.append(decoder_input_2.clone())

                get_idx = (decoder_input_2 >= self.voc.nwords).nonzero()
                decoder_input_2[get_idx] = self.voc.w2id['UNK']
                decoder_input_2 = self.embedding(decoder_input_2)

                decoder_input = torch.cat((decoder_input_1, decoder_input_2), 1) # batch_len x 2d
            ret_words = torch.stack(ret_words)

        # BECAUSE LAST STEP LOSS IS NOT COMPUTED!!!
        if is_validation:
            return self.loss.item()/(self.config.max_length-1), ret_words
        else:
            return 0.0, ret_words
