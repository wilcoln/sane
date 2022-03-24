import copy
import math
import random
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BartForSequenceClassification
from latent_rationale.nn.kuma_gate import KumaGate

class PriorKSModel(nn.Module):

    def __init__(self,
                 args):
        super().__init__()
        self.args = args
        self.uniform_prior = args.uniform_prior
        if not self.uniform_prior:
            self.bart_model = BartForSequenceClassification.from_pretrained('bart-base', output_hidden_states=True)
        
        self.z_layer = KumaGate(self.encoder.hidden_size)


    def get_prob_z_given_H(self, encoder_outputs, effects=None):
        '''
        '''
        z_dist = self.z_layer(encoder_outputs[0])

        # we sample once since the state was already repeated num_samples
        if self.training:
                z = z_dist.sample()  # [B, M, 1]
        else:
            # deterministic strategy
            p0 = z_dist.pdf(h.new_zeros(()))
            p1 = z_dist.pdf(h.new_ones(()))
            pc = 1. - p0 - p1  # prob. of sampling a continuous value [B, M]
            z = torch.where(p0 > p1, h.new_zeros([1]),  h.new_ones([1]))
            z = torch.where((pc > p0) & (pc > p1), z_dist.mean(), z)  # [B, M, 1]

        # mask invalid positions
        z = z.squeeze(-1)
        z = torch.where(mask, z, z.new_zeros([1]))

        self.z = z  # [B, T]
        self.z_dists = [z_dist]
        

        return z_dist, z

    def sample(self, dist_over_z):
        '''
        :param dist_over_z: B,prior_size
        :return: action, logprob of chosen action
        '''
        dist: torch.distributions.Categorical = torch.distributions.Categorical(probs=dist_over_z)
        action_idx = dist.sample()  # B
        return action_idx, dist.log_prob(action_idx)  #B;B

    def entropy(self, dist_over_z):
        '''
        :param dist_over_z: B,prior_size
        :return: entropy
        '''
        dist: torch.distributions.Categorical = torch.distributions.Categorical(probs=dist_over_z)
        entropy = dist.entropy()  # B
        return entropy.mean() # 1

