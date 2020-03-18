import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from src.models.attention import *

class PointerGen(nn.Module):
    def __init__(self, attn_model, embedding, cell_type, hidden_size, output_size, nlayers=1, dropout=0.1):
        super(PointerGen, self).__init__()

        # Keep for reference
        self.attn_model         = 'concat'
        self.hidden_size        = hidden_size
        self.output_size        = output_size
        self.nlayers            = nlayers
        self.dropout            = dropout
        self.cell_type          = cell_type

        # Define layers
        self.embedding = embedding
        self.embedding_size  = self.hidden_size + self.embedding.embedding_dim
        self.embedding_dropout = nn.Dropout(self.dropout)
        if self.cell_type == 'gru':
            self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.nlayers, dropout=(0 if self.nlayers == 1 else self.dropout))
        else:
            self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.nlayers, dropout=(0 if self.nlayers == 1 else self.dropout))
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.pgen_linear = nn.Linear(self.hidden_size * self.nlayers + self.hidden_size + self.embedding_size, 1)
        self.sig_hid = nn.Linear(self.hidden_size*self.nlayers, 1)

        self.attn = Attn(self.attn_model, self.hidden_size)

    def pval(self, hidden, context, decoder_emb):
        pval_in = torch.cat((hidden, context, decoder_emb), 1)
        pgen =  F.gelu(self.pgen_linear(pval_in))
        return pgen

    def forward(self, input_step, last_hidden, encoder_outputs, enc_batch_extend_vocab, extra_zeros, attn_masks, cov_vec = None):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word

        # input_step : [batch_size, embedding_size]
        # last_hidden : [layers, batchsize, hidden_dim]
        # encoder_outputs : [max_length, batchsize, hidden_dim]
        # attn_masks : [batchsize, max_length]
        embedded = self.embedding_dropout(input_step)
        embedded = embedded.view(1, input_step.size(0), self.embedding_size)

        rnn_output, hidden = self.rnn(embedded, last_hidden) # rnn_output : [1, batchsize, hidden_dim], hidden : [layers, batchsize, hidden_dim]

        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs, attn_masks, cov_vec) #  attn_weights : [batchsize, 1, max_src_length]

        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # context: [batchsize, 1, hidden_size]

        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0) # rnn_output : [batchsize, hidden_dim]
        hidden_inp = last_hidden.transpose(0, 1).contiguous().view(input_step.size(0), -1) # hidden : [batchsize, hidden_dim]
        context = context.squeeze(1) # context : [batchsize, hidden_dim]
        concat_input = torch.cat((rnn_output, context), 1) # concat_input : [batchsize, hidden_dim*2]
        pgen = F.sigmoid(self.pval(hidden_inp, context, embedded.squeeze(0)))

        concat_output = F.gelu(self.concat(concat_input)) # concat_output : [batchsize, hidden_dim]
        # Predict next word using Luong eq. 6
        output = self.out(concat_output) # output : [batchsize, output_dim (vocab size)]

        output_vocab_dist = F.softmax(output, dim=1)
        if extra_zeros is not None:
            output_vocab_dist = torch.cat([output_vocab_dist, extra_zeros], 1)
        # Calculate new probability for extended vocab using See. eq. 9
        output_vocab_dist = pgen * output_vocab_dist
        attn_dist = (1. - pgen) * attn_weights.squeeze()   # output : [batchsize, output_dim (vocab size)]
        output = output_vocab_dist.scatter_add(1, enc_batch_extend_vocab.transpose(0,1), attn_dist)
        output = torch.log(output + 10e-6)

        output = F.log_softmax(output, dim = 1)
        # Return output and final hidden state
        attn_weights = attn_weights.squeeze(1).transpose(0, 1) # attn_weight : [max_length, batchsize]

        hid_for_sig = hidden.transpose(0,1).contiguous()
        sig_hid = self.sig_hid(hid_for_sig.reshape(hidden.size()[1], hidden.size()[0]*hidden.size()[2]))

        return output, hidden, attn_weights, sig_hid
