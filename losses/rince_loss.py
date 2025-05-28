
'''
Proposed in this paper: https://arxiv.org/abs/2201.04309
Adapt from: https://github.com/CoinCheung/DenseCL/blob/722d75cb32fa69b885212ea7efc10786f916b582/densecl/rince.py#L78

This loss is not always greater than 0, and maybe we should add warmup to stablize training.

Sadly, if we use this to train denseCL from scratch, the loss would become nan if we use identical training parameters as original implementation(tuned for using info-nce). From my observation, the loss generates fierce gradient thus the model output logits becomes nan.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp



class RINCE(nn.Module):

    def __init__(self, q=0.5, lam=0.025):
        super(RINCE, self).__init__()
        self.q = q
        self.lam = lam

    def forward(self, pos, neg):
        loss = RINCEFunc.apply(pos, neg, self.lam, self.q)
        return loss.mean()


class RINCEFunc(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, pos, neg, lam, q):
        div_q = 1./q
        exp_pos = pos.exp().squeeze(1)
        exp_sum = exp_pos + neg.exp()  #.sum(dim=1)
        term1 = exp_pos.pow(q).neg_()
        term2 = exp_sum.mul_(lam).pow_(q)
        loss = (term1 + term2).mul_(div_q)

        ctx.vars = pos, neg, lam, q

        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        pos, neg, lam, q = ctx.vars
        exp_pos = pos.exp().squeeze(1)
        exp_neg = neg.exp()
        exp_sum = exp_pos + exp_neg #.sum(dim=1)

        d_pos = exp_sum.mul(lam).pow(q-1.).mul(lam).mul(exp_pos) - exp_pos.pow(q)
        d_neg = exp_sum.mul(lam).pow(q-1.).mul(lam).mul(exp_neg) #.unsqueeze(1)

        d_pos = d_pos.mul(grad_output) #.unsqueeze(1)
        d_neg = d_neg.mul(grad_output) #.unsqueeze(1)

        return d_pos, d_neg, None, None


class RINCEV2(nn.Module):

    def __init__(self, q=0.5, lam=0.025):
        super(RINCEV2, self).__init__()
        self.q = q
        self.div_q = 1./q
        self.lam = lam

    def forward(self, pos, neg):
        pos = pos.float()
        neg = neg.float()
        exp_pos = pos.exp() #.squeeze(1)
        exp_sum = exp_pos + neg.exp() #.sum(dim=1)

        term1 = exp_pos.pow(self.q).neg()
        term2 = exp_sum.mul(self.lam).pow(self.q)
        loss = (term1 + term2).mul(self.div_q)
        return loss.mean()

