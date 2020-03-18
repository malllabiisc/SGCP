import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(nn.init.xavier_uniform(torch.FloatTensor(1, hidden_size)))
            self.w_c = nn.Parameter(nn.init.xavier_uniform(torch.FloatTensor(1, hidden_size)))

    def dot_score(self, hidden, encoder_outputs):
        return torch.sum(hidden * encoder_outputs, dim=2)

    def general_score(self, hidden, encoder_outputs):
        energy = self.attn(encoder_outputs)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_outputs, cov_vec):
        if cov_vec is not None:
            cov_mech = self.w_c.squeeze() * cov_vec.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            energy = (self.attn(torch.cat((hidden.expand(encoder_outputs.size(0), -1, -1), encoder_outputs), 2)) + cov_mech).tanh()
        else:
            energy = self.attn(torch.cat((hidden.expand(encoder_outputs.size(0), -1, -1), encoder_outputs), 2)).tanh()

        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs, attn_masks, cov_vec = None):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs, cov_vec)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        attn_energies.masked_fill_(attn_masks, -np.inf)
        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
