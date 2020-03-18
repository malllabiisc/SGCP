import torch
import torch.nn as nn
import numpy as np
import sys

import torch.nn.functional as F
from src.models.attention import *
from src.models.pointergen import *

class TreeEncoder(nn.Module):

    def __init__(self, config, embedding, voc, device):
        super(TreeEncoder, self).__init__()
        self.config = config
        self.voc = voc
        self.embedding = embedding
        self.transform = nn.Linear(self.config.emb_size + self.config.hidden_size, self.config.hidden_size)
        # self.nonlinearity = F.GeLU()
        self.device = device
        self.leaf_forward    = PointerGen(self.config.attn_type,
                                          self.embedding,
                                          self.config.cell_type,
                                          self.config.hidden_size,
                                          self.voc.nwords,
                                          self.config.depth,
                                          self.config.s2sdprate).to(device)

    def nonleaf_forward(self, decoder_input):
        output = self.transform(decoder_input)
        return F.gelu(output)

    def all_stacks_empty(self, tree_stacks):
        return (np.sum([len(stk) for stk in tree_stacks]) == 0)

    def forward(self, encoder_outputs, tgt_trees):

        # encoder outputs are batch_len x d
        batch_size = encoder_outputs.size()[0]
        # pdb.set_trace()
        tree_stacks = [[['ROOT', 0]] for _ in range(batch_size)]
        root_embed = torch.LongTensor([self.voc.w2id['ROOT'] for i in range(batch_size)]).to(self.device) # batch_len x 1
        root_embed = self.embedding(root_embed) # batch_len x d
        initial_input = torch.cat((encoder_outputs, root_embed), 1) # batch_len x 2d
        # May be we can ignore encoder outputs at this point and start with root embedding itself
        embedding_stacks = [[embedding] for embedding in self.nonleaf_forward(initial_input)] # initial input is batch_len x 2d (encoder output | 'ROOT' embedding)
        leaf_embedding_list = [[] for _ in range(batch_size)]

        while not self.all_stacks_empty(tree_stacks):

            # Construct input for next step of non-leaf decoder
            next_input_parent = []
            next_input_child = []

            # Operate on each stack iteratively
            for i, stk in enumerate(tree_stacks):
                cur_tree = tgt_trees[i]
                if len(stk) > 0:
                    stk_top = stk[-1]
                    cur_node = stk_top[0]
                    next_child = stk_top[1]

                # Go to parent which has next child
                while len(stk) > 0 and next_child >= len(cur_tree[cur_node]):
                    if next_child == 0:
                        leaf_embedding_list[i].append(embedding_stacks[i][-1])
                    stk.pop()
                    embedding_stacks[i].pop()
                    if len(stk) > 0:
                        stk_top = stk[-1]
                        cur_node = stk_top[0]
                        next_child = stk_top[1]

                if len(stk) > 0:
                    stk_top = stk[-1]
                    cur_node = stk_top[0]
                    next_child = stk_top[1]
                    next_input_parent.append(embedding_stacks[i][-1])
                    next_input_child.append(self.voc.w2id[cur_tree[cur_node][next_child].split('-')[0]])
                    stk[-1][1] += 1
                    stk.append([cur_tree[cur_node][next_child], 0])
                else:
                    # These are only for batch processing purposes and won't effect any parameter updates as theses will never end up in leaf embedding list
                    null_embedding = torch.FloatTensor([0 for _ in range(self.config.hidden_size)]).to(self.device)
                    next_input_parent.append(null_embedding)
                    next_input_child.append(self.voc.w2id['UNK'])

            next_input_child = torch.LongTensor(next_input_child).to(self.device) # batch_len x 1
            next_input_child = self.embedding(next_input_child) # batch_len x d
            next_input_parent = torch.stack(next_input_parent) # batch_len x d

            next_input = torch.cat((next_input_parent, next_input_child), 1) #batch_len x 2d
            decoder_outputs = self.nonleaf_forward(next_input) #batch_len x d

            # Due to this step, tree stack and embedding stack size must be same at the beginning of every iteration
            for i, stk in enumerate(embedding_stacks):
                if len(stk) > 0:
                    embedding_stacks[i].append(decoder_outputs[i])

        return leaf_embedding_list
