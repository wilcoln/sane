from transformers import RobertaForSequenceClassification

import torch
import torch.nn as nn
import torch.nn.functional as F
from rationales import BartModel
from prior import PriorKSModel

from comet2.comet_model import PretrainedCometModel

ALPHA = 0.4

class GenerationKSModel(nn.Module):
    def __init__(self,
                 args):
        super().__init__()

        self.args = args
        self.prior_model = PriorKSModel(args)

        self.bart_model = BartModel(args)
        self.criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def forward(
            self,
            input_ids,
            lm_labels=None,
            generate=False,
            knowledge=None,
            **kwargs):
        '''
        '''

        sampler_model = self.prior_model

        if not generate:

            z_given_h_and_x = sampler_model.get_prob_z_given_H_and_x(input_ids)  
            z_given_h = self.prior_model.get_prob_z_given_H(knowledge, input_ids)

            log_probs_lm = []

            
            z_dist, z = sampler_model.sample(z_given_h_and_x) 

            input_ids = input_ids * z
            lm_logits, *_ = self.bart_model(
                input_ids,
            )

            # LM
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            num_labels = (lm_labels != -100).sum([-3, -2, -1])  # B
            ll_lm = -1 * self.criterion_lm(lm_logits_flat_shifted, lm_labels_flat_shifted)  # B x C x T
            ll_lm = ll_lm.view(lm_labels.size(0), -1).sum(1)  # B

            log_prob_x_z_given_h = ll_lm
            log_probs_lm.append(log_prob_x_z_given_h / num_labels) 

            return lm_logits


class RExC(nn.Module):
    def __init__(self,
                 args):
        super().__init__()

        self.args = args
        self.rationale = BartModel(args)
        self.cs = PretrainedCometModel(args)
        self.gen_model = GenerationKSModel(args)

    def forward(self, input_ids, labels, expl_labels):

        r_loss, z = self.rationale(input_ids, labels)
        c_output = self.cs(input_ids, z)
        g_loss = self.gen_model(input_ids, c_output, expl_labels)

        return ALPHA*r_loss + (1-ALPHA)*g_loss