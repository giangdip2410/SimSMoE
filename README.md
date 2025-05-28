# SimSMoE: Toward Efficient Training Mixture of Experts via Solving Representational Collapse

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repo contains the code for this paper [SimSMoE: Toward Efficient Training Mixture of Experts via Solving Representational Collapse](https://www.arxiv.org/pdf/2406.15883)

Giang Do, Hung Le, Truyen Tran

## Overview

Sparse mixture of experts (SMoE) have emerged as an effective approach for scaling large language models while keeping a constant computational cost. Regardless of several notable successes of SMoE, effective training such architecture remains elusive due to the representation collapse problem, which in turn harms model performance and causes parameter redundancy. In this work, we present Similarity-based Sparse Mixture of Experts (SimSMoE), a novel similarity of neural network algorithm, that guarantees a solution to address the representation collapse issue between experts given a fixed FLOPs budget. We conduct extensive empirical evaluations on three large language models for both Pre-training and Fine-tuning tasks to illustrate the efficacy, robustness, and scalability of our method. The results demonstrate that SimSMoE significantly enhances existing routing policy and outperforms other SMoE routing methods in performance for the tasks.

## Prerequisites
- [FastMoE](https://github.com/laekov/fastmoe): A fast MoE impl for PyTorch

## Running Experiments in the Paper

#### Pre-training
- Download the enwik8, text8, wikitext-2 dataset from [here](https://github.com/laekov/fastmoe/blob/master/examples/transformer-xl/scripts/getdata.sh), then change bash scripts based on your local data paths`</br>
```bash
data_folder/
└── pretraining
    └── enwik8
        ├── test.txt
        ├── train.txt
        └── valid.txt
    └── text8
        ├── test.txt
        ├── train.txt
        └── valid.txt
    └── wikitext-2
        ├── test.txt
        ├── train.txt
        └── valid.txt
```

- Select the Transformer architecture, its scale, and the type of SMoE layer. We support:

|                     | SMoE | SMoE-Dropout | XMoE | StableMoE | SimSMoE     |
|---------------------|------|--------------|------|-----------|-------------|
| Transformer (S/M/L) |  ✅  |     ✅       |  ✅  |     ✅    |    ✅      |
| GLAM (S/M/L)        |  ✅  |     ✅       |  ✅  |     ✅    |    ✅      |

- Run all corresponding scripts: </br>
`bash enwik8_exp.sh`
`bash text8_exp.sh`
`bash wikitext2_exp.sh`

- The checkpoint will be saved at `checkpoints/enwik8/transformers-s` during training. 

## Citation

@inproceedings{do-etal-2025-simsmoe,
    title = "{S}im{SM}o{E}: Toward Efficient Training Mixture of Experts via Solving Representational Collapse",
    author = "Do, Giang  and
      Le, Hung  and
      Tran, Truyen",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-naacl.107/",
    pages = "2012--2025",
    ISBN = "979-8-89176-195-7",
    abstract = "Sparse mixture of experts (SMoE) have emerged as an effective approach for scaling large language models while keeping a constant computational cost. Regardless of several notable successes of SMoE, effective training such architecture remains elusive due to the representation collapse problem, which in turn harms model performance and causes parameter redundancy. In this work, we present Similarity-based Sparse Mixture of Experts (SimSMoE), a novel similarity of neural network algorithm, that guarantees a solution to address the representation collapse issue between experts given a fixed FLOPs budget. We conduct extensive empirical evaluations on three large language models for both Pre-training and Fine-tuning tasks to illustrate the efficacy, robustness, and scalability of our method. The results demonstrate that SimSMoE significantly enhances existing routing policy and outperforms other SMoE routing methods in performance for the tasks. Our implementation is publicly available at https://github.com/giangdip2410/SimSMoE."
}