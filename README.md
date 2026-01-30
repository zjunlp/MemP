<h1 align="center"> MemP Exploring Agent Procedural Memory </h1>


<p align="center">
  <a href="https://arxiv.org/pdf/2508.06433" target="_blank">📄arXiv</a> •
  <a href="https://huggingface.co/papers/2508.06433" target="_blank">🤗HFPaper</a> 
</p>

<!-- [![Awesome](https://awesome.re/badge.svg)](https://github.com/zjunlp/WorFBench)  -->
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/zjunlp/SynWorld?color=green) 

## Table of Contents

- 🌻[Acknowledgement](#acknowledgement)
- 🌟[Overview](#overview)
- 🔧[Installation](#installation)
- ✏️[Offline Running](#offline-running)
- 📝[Online Running](#online-running)
- 🚩[Citation](#citation)
<!-- - 🎉[Contributors](#🎉contributors) -->

---

## 🌻Acknowledgement

Our code is referenced and adapted from [Langchain](https://www.langchain.com/), [ETO](https://github.com/Yifan-Song793/ETO?tab=readme-ov-file). And Thanks to [ETO](https://github.com/Yifan-Song793/ETO?tab=readme-ov-file) provide the trajectory on train set.


## 🌟Overview

Large Language Models based agents excel at diverse tasks yet they suffer from brittle procedural memory that is manually engineered or entangled in static parameters. In this work we investigate strategies to endow agents with a learnable updatable and lifelong procedural memory. We propose MemP that distills past agent trajectories into both fine grained step by step instructions and higher level script like abstractions and explore the impact of different strategies for Build Retrieval and Update of procedural memory. Coupled with a dynamic regimen that continuously updates corrects and deprecates its contents this repository evolves in lockstep with new experience. Empirical evaluation on TravelPlanner and ALFWorld shows that as the memory repository is refined agents achieve steadily higher success rates and greater efficiency on analogous tasks. Moreover procedural memory built from a stronger model retains its value migrating the procedural memory to a weaker model can also yield substantial performance gains.


In MemP, we support two strategies for building procedural memory: one constructs procedural memory **offline** using existing trajectories, and the other adopts a self-learning approach, starting from scratch to execute agent tasks **online** while actively learning procedural memory.
## 🔧Installation

```bash
git clone https://github.com/zjunlp/MemP
cd ProceduralMem
pip install -r requirements.txt

```

After installed, init neccessary  Environment Variables

```bash
export OPENAI_API_KEY=YOUR_API_KEY
export OPENAI_API_BASE=YOUR_API_BASE_URL
export EMBEDDING_MODEL_KEY=YOUR_EMBEDDING_MODEL_KEY
export EMBEDDING_MODEL_BASE_URL=YOUR_EMBEDDING_MODEL_BASE_URL
```

## ✏️Offline Running
```bash
python run_memp_offline.py \
    --model your_model_name \
    --split dev_or_test \
    --batch_size concurrency_num \
    --max_steps n \
    --exp_name save_name \
    --few_shot \
    --use_memory
```



## 📝Online Running
```bash
python run_memp_online.py \
    --model your_model_name \
    --split dev_or_test \
    --batch_size concurrency_num \
    --max_steps n \
    --exp_name save_name \
    --few_shot \
    --use_memory \
    --overwrite
```



## 🚩Citation

If this work is helpful, please kindly cite as:

```bibtex
@article{DBLP:journals/corr/abs-2508-06433,
  author       = {Runnan Fang and
                  Yuan Liang and
                  Xiaobin Wang and
                  Jialong Wu and
                  Shuofei Qiao and
                  Pengjun Xie and
                  Fei Huang and
                  Huajun Chen and
                  Ningyu Zhang},
  title        = {Memp: Exploring Agent Procedural Memory},
  journal      = {CoRR},
  volume       = {abs/2508.06433},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2508.06433},
  doi          = {10.48550/ARXIV.2508.06433},
  eprinttype    = {arXiv},
  eprint       = {2508.06433},
  timestamp    = {Sat, 13 Sep 2025 14:46:20 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2508-06433.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```