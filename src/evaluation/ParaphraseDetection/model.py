import torch
import torch.nn as nn
from pytorch_transformers import RobertaModel
import ipdb as pdb

class RoBERTaPPD(nn.Module):
    
    def __init__(self):
        super(RoBERTaPPD, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.cls = nn.Linear(768, 1)

    def forward(self, x, attn_masks):
        cls_feat = self.roberta(x, attention_mask = attn_masks)
        try:
            cls_feat = cls_feat[1]
        except:
            pdb.set_trace()
        out = self.cls(cls_feat)

        return out
