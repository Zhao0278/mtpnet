# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, code_inputs, nl_inputs, return_vec=False, return_scores=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        encoder_output = self.encoder(inputs, attention_mask=inputs.ne(1))
        outputs = encoder_output[1]

        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]

        if return_vec:
            return code_vec, nl_vec
        scores = (nl_vec[:, None, :] * code_vec[None, :, :]).sum(-1)
        if return_scores:
            return scores
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(bs, device=scores.device))
        return loss, code_vec, nl_vec

    def feature(self, code_inputs, nl_inputs):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        encoder_output = self.encoder(inputs, attention_mask=inputs.ne(1))
        code_feature = encoder_output.pooler_output[:bs]
        nl_feature = encoder_output.pooler_output[bs:]
        return code_feature, nl_feature