import os, sys
import argparse
import math, random
import torch
import tqdm

PARAMS_CONFIG = {
    # env-specific
    "env_params": {
        "--distributed": {
            "action": "store_true",
            "default": False,
            "help": "enable distributed training."
            "(otherwise will use all available GPUs with dataparallel)",
            "dest": "distributed",
        },
        "--local_rank": {
            "type": int,
            "default": 0,
            "help": "used in distributed training",
            "dest": "local_rank",
        },
    },
    # data-specific
    "data_params": {
        "--data": {
            "type": str,
            "default": "data/text8",
            "help": "data location " "(must contain train.txt, valid.txt and test.txt)",
            "dest": "data_path",
        },
    },
    # model-specific
    "model_params": {
        "--hid-sz": {
            "type": int,
            "default": 256,
            "help": "hidden size (i.e. model size)",
            "dest": "hidden_size",
        },
        "--inner-hid-sz": {
            "type": int,
            "default": 1024,
            "help": "inner hidden size of FF layer",
            "dest": "inner_hidden_size",
        },
        "--nlayers": {
            "type": int,
            "default": 8,
            "help": "number of layers",
            "dest": "nb_layers",
        },
        "--block-sz": {
            "type": int,
            "default": 64,
            "help": "block size " "(the length of sequence to process in parallel)",
            "dest": "block_size",
        },
        "--nheads": {
            "type": int,
            "default": 2,
            "help": "number of self-attention heads",
            "dest": "nb_heads",
        },
        "--attn-span": {
            "type": int,
            "default": 32,
            "help": "length of the attention span",
            "dest": "attn_span",
        },
        "--dropout": {
            "type": float,
            "default": 0.2,
            "help": "dropout rate of ReLU and attention",
            "dest": "dropout",
        },
        "--architecture": {
            "type": str,
            "default": None,
            "help": "arch",
            "dest": "architecture",
        },
        "--base_arch": {
            "type": str,
            "default": None,
            "help": "arch",
            "dest": "base_arch",
        },
        "--smoe_dropout": {
            "action": "store_true",
            "default": False,
            "help": "enable SMoE-drop - Freeze gate",
            "dest": "smoe_dropout",
        },
        "--top_k": {
            "type": int,
            "default": 2,
            "help": "Number of activate experts",
            "dest": "top_k",
        },
        "--num_expert": {
            "type": int,
            "default": 16,
            "help": "Number of experts",
            "dest": "num_expert",
        },
        "--contrastive": {
            "action": "store_true",
            "default": False,
            "help": "enable Contrastive mode",
            "dest": "contrastive",
        },
        "--cont_freq": {
            "type": float,
            "default": 1.0,
            "help": "Frequency of Contrastive Activate",
            "dest": "cont_freq",
        },
        "--sim_threshold": {
            "type": float,
            "default": 0.5,
            "help": "Threshold to classfy collapse or not",
            "dest": "sim_threshold",
        },
        "--contrative_rate": {
            "type": float,
            "default": 1.0,
            "help": "Rate impact to contrative loss",
            "dest": "contrative_rate",
        },
        "--contrative_loss": {
            "type": str,
            "default": "cka",
            "help": "Name of Contrative Loss: cka, absconsine, consine, dcl, directclr, hypersphere, infonce, rincev1, rincev2, simclrv1, simclrv2, tico",
            "dest": "contrative_loss",
        },
        "--cka_mode": {
            "type": str,
            "default": "kernel",
            "help": "Mode for CKA loss",
            "dest": "cka_mode",
        },
        "--sigma": {
            "type": float,
            "default": 0.8,
            "help": "sigma for CKA Kernel loss ",
            "dest": "sigma",
        },
        "--load_balance": {
            "type": float,
            "default": 0.0,
            "help": "load_balance",
            "dest": "load_balance",
        },
        "--gate_name": {
            "type": str,
            "default": "smoe",
            "help": "gate_name",
            "dest": "gate_name",
        },
    },
    # optimization-specific
    "optim_params": {
        "--lr": {"type": float, "default": 0.03, "help": "learning rate", "dest": "lr"},
        "--momentum": {
            "type": float,
            "default": 0.9,
            "help": "SGD momentum",
            "dest": "momentum",
        },
        "--optim": {
            "type": str,
            "default": "sgd",
            "help": "optimization method: sgd | adagrad",
            "dest": "optim",
        },
        "--lr-warmup": {
            "type": int,
            "default": 0,
            "help": "linearly increase LR from 0 " "during first lr_warmup updates",
            "dest": "lr_warmup",
        },
        "--grad-clip": {
            "type": float,
            "default": 0,
            "help": "[only works with adagrad!] "
            "clip gradient of each module parameters by a given "
            "value",
            "dest": "grad_clip",
        },
    },
    # trainer-specific
    "trainer_params": {
        "--batch-sz": {
            "type": int,
            "default": 64,
            "help": "batch size",
            "dest": "batch_size",
        },
        "--batch-split": {
            "type": int,
            "default": 1,
            "help": "split a batch into smaller parts to fit in GPU memory",
            "dest": "batch_split",
        },
        "--nbatches": {
            "type": int,
            "default": 1000,
            "help": "number of batches in each iteration",
            "dest": "nb_batches_per_iter",
        },
        "--niter": {
            "type": int,
            "default": 1000,
            "help": "number of iterations to train",
            "dest": "nb_iter",
        },
        "--checkpoint": {
            "type": str,
            "default": "",
            "help": "path to save/load model",
            "dest": "checkpoint_path",
        },
        "--full-eval-mode": {
            "action": "store_true",
            "default": False,
            "help": "do evaluation on the whole validation and the test data",
            "dest": "full_eval_mode",
        },
    },
    # adaptive attention span specific params
    "adapt_span_params": {
        "--adapt-span": {
            "action": "store_true",
            "default": False,
            "help": "enable adaptive attention span",
            "dest": "adapt_span_enabled",
        },
        "--adapt-span-loss": {
            "type": float,
            "default": 0,
            "help": "the loss coefficient for span lengths",
            "dest": "adapt_span_loss",
        },
        "--adapt-span-ramp": {
            "type": int,
            "default": 32,
            "help": "ramp length of the soft masking function",
            "dest": "adapt_span_ramp",
        },
        "--adapt-span-init": {
            "type": float,
            "default": 0,
            "help": "initial attention span ratio",
            "dest": "adapt_span_init",
        },
        "--adapt-span-cache": {
            "action": "store_true",
            "default": False,
            "help": "adapt cache size as well to reduce memory usage",
            "dest": "adapt_span_cache",
        },
    },
}
