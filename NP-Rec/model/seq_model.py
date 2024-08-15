import copy
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from baseline.single_repr.modules import *


def clones(layer, depth):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(depth)])


class SubLayerConnect(nn.Module):
    def __init__(self, features, dropout_ratio):
        super(SubLayerConnect, self).__init__()
        self.norm = nn.LayerNorm(features)
        self.drop = nn.Dropout(p=dropout_ratio)

    def forward(self, x, sublayer):
        y = self.norm(x + self.drop(sublayer(x)))
        return y

class FFN(nn.Module):
    def __init__(self, input_dim, exp_factor, dropout_ratio):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, exp_factor * input_dim)
        self.fc2 = nn.Linear(exp_factor * input_dim, input_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.act(self.fc2(x))
        return x


class MambaBlock(nn.Module):
    def __init__(self, input_dim, exp_factor, dropout_ratio):
        super(MambaBlock, self).__init__()
        self.ssm_layer = Mamba(input_dim)
        self.ffn_layer = FFN(input_dim, exp_factor, dropout_ratio)
        self.sub_layer_1 = SubLayerConnect(input_dim, dropout_ratio)
        self.sub_layer_2 = SubLayerConnect(input_dim, dropout_ratio)

    def forward(self, x):
        y = self.sub_layer_1(x, self.ssm_layer)
        z = self.sub_layer_2(y, self.ffn_layer)
        return z


class SSM(nn.Module):
    def __init__(self, input_dim, exp_factor=4, dropout_ratio=0.3, depth=2):
        super(SSM, self).__init__()
        self.block = MambaBlock(input_dim, exp_factor, dropout_ratio)
        self.stack_blocks = clones(self.block, depth)

    def forward(self, x):
        for block in self.stack_blocks:
            x = block(x)
        return x


class SurRec(nn.Module):
    def __init__(self, mixer_type, max_len, session_len, d_model, hidden_drop, depth):
        super(SurRec, self).__init__()
        assert mixer_type in ['RNN', 'CNN', 'Transformer', 'MLP']

        self.mixer_type = mixer_type
        if mixer_type == 'RNN':
            self.model_name = 'GRU4Rec'
            self.mixer = RNNMixer(d_model, hidden_drop, depth)
        elif mixer_type == 'CNN':
            self.mixer = CNNMixer(d_model, d_model, 3, [1, 4] * depth, hidden_drop)
            self.model_name = 'NextItNet'
        elif mixer_type == 'Transformer':
            self.embed_pos = nn.Embedding(max_len, d_model)
            self.mixer = Transformer(d_model, 4, 4, hidden_drop, depth)
            self.model_name = 'SASRec'
        else:
            self.mixer = TriMixer(max_len, session_len)
            self.model_name = 'TriMLP'


    def forward(self, src_seq):
        if self.mixer_type is not 'Transformer':
            mixer_output = self.mixer(src_seq)
        else:
            mask = get_mask(src_seq, bidirectional=False)
            pos = self.embed_pos.weight[:src_seq.size(1), :]
            x = self.drop_embed(src_seq + pos)
            mixer_output = self.mixer(x, mask)

        return mixer_output
