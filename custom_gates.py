import os, sys
import argparse
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import pdb
import numpy as np
from fmoe.gates.base_gate import BaseGate
import random
from itertools import combinations

__all__ = [
    "CustomNaiveGate_Balance_SMoE",
    "CustomNaiveGate_Balance_XMoE",
    "CustomNaiveGate_Balance_StableMoE",
]


class CustomNaiveGate_Balance_SMoE(BaseGate):
    def __init__(
        self,
        d_model,
        num_expert,
        world_size,
        top_k=2,
        contrastive=False,
        cont_freq=1.0
    ):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.loss = None
        #check contrastive mode or not
        self.contrastive = contrastive
        if contrastive:
            self.trackable = True
            #pair topk
            self.pair = {} # N * (N-1) / 2  = number of key
            #combination experts
            exp_comb = list(combinations(list(range(num_expert)), top_k))
            #self.num_pair = len(exp_comb)
            #tracking collapse
            self.tracking = {} # N * (N-1) / 2  = number of key
            for idx1, i in enumerate(exp_comb):
                self.pair[idx1] =  i
                self.tracking[i] = 0.0
            #threah hold
            self.thread = cont_freq
        else:
            self.trackable = False

        

    def set_load_balance(self, gate, gate_top_k_idx):
        # gate_top_k_idx (tokens_number, top-k)
        # gate_top_k_val (tokens_number, top-k)

        score = F.softmax(gate, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss
    
    def update_tracking(self, idx):
        idx2 , _ = torch.sort(idx, dim=1)
        bs = idx2.shape[0]
        sing = list(map(tuple, idx2.cpu().numpy())) #[(i, j) for i, j in zip(sing1, sing2)]
        counter_expert = Counter(sing)
        #update tracking
        for k in counter_expert.keys():
            self.tracking[k] += (counter_expert[k] * 1.0 /bs)
    

    def pair_to_idex(self, idx2, pair):
        k, v = pair
        select_idx = (idx2[:, 0] == k) & (idx2[:, 1] == v) 
        return select_idx

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)
        bs = gate.shape[0]

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        elif self.contrastive and self.training:
            #update tracking
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            self.update_tracking(gate_top_k_idx)
            #get top value
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k
            #sortching index
            gate_top_k_idx_sort , _ = torch.sort(gate_top_k_idx,dim=1)
            #select pair
            pair_sel = {}
            for k, v in self.tracking.items():
                if v > self.thread:
                    tmp_select = self.pair_to_idex(gate_top_k_idx_sort, k)
                    #save to pair_sel
                    pair_sel[k] = tmp_select
                    #back selected key to zeros
                    self.tracking[k] = 0.0

        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            if self.trackable and self.training:
                #update tracking
                self.update_tracking(gate_top_k_idx)
            #get top value
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)

        self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        elif self.contrastive and self.training:
            return gate_top_k_idx, gate_score, pair_sel
        else:
            return gate_top_k_idx, gate_score


class CustomNaiveGate_Balance_XMoE(BaseGate):
    def __init__(
        self,
        d_model,
        num_expert,
        world_size,
        top_k=2,
        contrastive=False,
        cont_freq=1.0
    ):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.loss = None
        #check contrastive mode or not
        self.contrastive = contrastive
        if contrastive:
            self.trackable = True
            #pair topk
            self.pair = {} # N * (N-1) / 2  = number of key
            #combination experts
            exp_comb = list(combinations(list(range(num_expert)), top_k))
            #self.num_pair = len(exp_comb)
            #tracking collapse
            self.tracking = {} # N * (N-1) / 2  = number of key
            for idx1, i in enumerate(exp_comb):
                self.pair[idx1] =  i
                self.tracking[i] = 0.0
            #threah hold
            self.thread = cont_freq
        else:
            self.trackable = False



        expert_embeddings = torch.empty(num_expert, 8)
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

        self.inp_reduction = torch.nn.Linear(d_model, 8, bias=False)

    def update_tracking(self, idx):
        idx2 , _ = torch.sort(idx, dim=1)
        bs = idx2.shape[0]
        sing = list(map(tuple, idx2.cpu().numpy())) #[(i, j) for i, j in zip(sing1, sing2)]
        counter_expert = Counter(sing)
        #update tracking
        for k in counter_expert.keys():
            self.tracking[k] += (counter_expert[k] * 1.0 /bs)
    

    def pair_to_idex(self, idx2, pair):
        k, v = pair
        select_idx = (idx2[:, 0] == k) & (idx2[:, 1] == v) 
        return select_idx

    
    
    
    def set_load_balance(self, gate, gate_top_k_idx):
        # gate_top_k_idx (tokens_number, top-k)
        # gate_top_k_val (tokens_number, top-k)

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, return_all_scores=False):

        reduced_inp = self.inp_reduction(inp)
        with torch.no_grad():
            expert_embeddings_norm = self.expert_embeddings.norm(
                p=2.0, dim=1, keepdim=True
            )
            self.expert_embeddings.mul_(1.5 / expert_embeddings_norm)

        gate = self._cosine(reduced_inp, self.expert_embeddings)
        gate = self._make_finite(gate)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        
        elif self.contrastive and self.training:
            #update tracking
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            self.update_tracking(gate_top_k_idx)
            #get top value
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k
            #sortching index
            gate_top_k_idx_sort , _ = torch.sort(gate_top_k_idx,dim=1)
            #select pair
            pair_sel = {}
            for k, v in self.tracking.items():
                if v > self.thread:
                    tmp_select = self.pair_to_idex(gate_top_k_idx_sort, k)
                    #save to pair_sel
                    pair_sel[k] = tmp_select
                    #back selected key to zeros
                    self.tracking[k] = 0.0
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            if self.trackable and self.training:
                #update tracking
                self.update_tracking(gate_top_k_idx)
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)

        self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        elif self.contrastive and self.training:
            return gate_top_k_idx, gate_score, pair_sel
        else:
            return gate_top_k_idx, gate_score

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores


class CustomNaiveGate_Balance_StableMoE(BaseGate):
    r"""
    Naive Gate StableMoE
    """

    def __init__(
        self,
        d_model,
        num_expert,
        world_size,
        top_k=2,
        contrastive=False,
        cont_freq=1.0
    ):
        super().__init__(num_expert, world_size)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.loss = None
        #check contrastive mode or not
        self.contrastive = contrastive
        if contrastive:
            self.trackable = True
            #pair topk
            self.pair = {} # N * (N-1) / 2  = number of key
            #combination experts
            exp_comb = list(combinations(list(range(num_expert)), top_k))
            #self.num_pair = len(exp_comb)
            #tracking collapse
            self.tracking = {} # N * (N-1) / 2  = number of key
            for idx1, i in enumerate(exp_comb):
                self.pair[idx1] =  i
                self.tracking[i] = 0.0
            #threah hold
            self.thread = cont_freq
        else:
            self.trackable = False


        expert_embeddings = torch.empty(num_expert, d_model)
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

    def update_tracking(self, idx):
        idx2 , _ = torch.sort(idx, dim=1)
        bs = idx2.shape[0]
        sing = list(map(tuple, idx2.cpu().numpy())) #[(i, j) for i, j in zip(sing1, sing2)]
        counter_expert = Counter(sing)
        #update tracking
        for k in counter_expert.keys():
            self.tracking[k] += (counter_expert[k] * 1.0 /bs)
    

    def pair_to_idex(self, idx2, pair):
        k, v = pair
        select_idx = (idx2[:, 0] == k) & (idx2[:, 1] == v) 
        return select_idx


    def set_load_balance(self, gate, gate_top_k_idx):
        # gate_top_k_idx (tokens_number, top-k)
        # gate_top_k_val (tokens_number, top-k)

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, return_all_scores=False):

        gate = self._cosine(inp, self.expert_embeddings)
        gate = self._make_finite(gate)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        

        elif self.contrastive and self.training:
            #update tracking
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            self.update_tracking(gate_top_k_idx)
            #get top value
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k
            #sortching index
            gate_top_k_idx_sort , _ = torch.sort(gate_top_k_idx,dim=1)
            #select pair
            pair_sel = {}
            for k, v in self.tracking.items():
                if v > self.thread:
                    tmp_select = self.pair_to_idex(gate_top_k_idx_sort, k)
                    #save to pair_sel
                    pair_sel[k] = tmp_select
                    #back selected key to zeros
                    self.tracking[k] = 0.0
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            if self.trackable and self.training:
                #update tracking
                self.update_tracking(gate_top_k_idx)
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)

        self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        elif self.contrastive and self.training:
            return gate_top_k_idx, gate_score, pair_sel
        else:
            return gate_top_k_idx, gate_score

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores
