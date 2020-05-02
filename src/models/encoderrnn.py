import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_sparse import spmm, transpose
# from scipy.linalg import block_diag
# from scipy.sparse import csr_matrix
import pdb



class EncoderRNN(nn.Module):
    '''
    EncoderRNN helps in building the sentence encoding module for a batched version
    of data that is sent in [T x B] having corresponding input lengths in [1 x B]

    Args:
        hidden_size: Hidden size of the RNN cell
        embedding: Embeddings matrix [vocab_size, embedding_dim]
        cell_type: Type of RNN cell to be used : LSTM, GRU
        nlayers: Number of layers of LSTM (default = 1)
        dropout: Dropout Rate (default = 0.1)
        bidirectional: Bidirectional model to be formed (default: False)
    '''
    def __init__(self, hidden_size, embedding, cell_type, nlayers=1, dropout=0.1, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout
        self.embedding = embedding
        self.cell_type = cell_type
        self.embedding_size = self.embedding.embedding_dim
        self.bidirectional = bidirectional

        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(self.embedding_size, self.hidden_size,
                               num_layers=self.nlayers, dropout = (0 if self.nlayers == 1 else dropout),
                               bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(self.embedding_size, self.hidden_size,
                              num_layers=self.nlayers, dropout=(0 if self.nlayers == 1 else dropout),
                              bidirectional=bidirectional)

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs

        return outputs, hidden
