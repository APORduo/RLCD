# RLCD: Generalized Category Discovery via Reciprocal Learning and Class-Wise Distribution Regularization

This repository contains the official implementation of our ICML 2025 paper:

**Generalized Category Discovery via Reciprocal Learning and Class-Wise Distribution Regularization**  
*Duo Liu, Zhiquan Tan, Linglan Zhao, Zhongqiang Zhang, Xiangzhong Fang, Weiran Huang*  


---

## 📖 Overview
Generalized Category Discovery (GCD) aims to identify unlabeled samples by leveraging base knowledge from labeled ones, where the unlabeled set consists of both base and novel classes.  
Our method, **RLCD**, introduces:
- **Reciprocal Learning Framework (RLF):** An auxiliary branch devoted to base classification, forming a virtuous cycle with the main branch.  
- **Class-Wise Distribution Regularization (CDR):** Mitigates bias towards base classes and boosts novel class performance.  

Together, RLCD achieves superior performance across multiple GCD benchmarks with negligible extra computation.

## 📂 Dataset Preparation
Download and preprocess datasets following instructions in [SimGCD](https://github.com/CVMI-Lab/SimGCD).


## 🚀 Quick Start
```bash
python train.py -d cifar100     -m 2.0 --fp16 --cross_w 0.5 --dw 0.5  --difflr
python train.py -d cub          -m 2.0 --fp16 --cross_w 0.5 --dw 0.5  --difflr
python train.py -d scars        -m 2.0 --fp16 --cross_w 0.5 --dw 1.0
python train.py -d herbarium_19 -m 2.0 --fp16 --cross_w 0.5 --dw 1.0 --fc_temp 0.07 --eval_funcs v2 v2b
```


## ⚙️ Hyperparameters and Tuning Tips

As reported in the paper, the **class-wise distribution regularization weight** `dw` and the **distillation loss weight** `cross_w` are **not heavily tuned**.

In practice, we recommend:

- Starting from the default configurations provided above;
- Adjusting `dw` and `cross_w` based on evaluation performance;


## 📜 Citation

If you find this repository or our paper helpful in your research, please consider citing:

```bibtex
@inproceedings{liu2025rlcd,
  title={Generalized Category Discovery via Reciprocal Learning and Class-Wise Distribution Regularization},
  author={Liu, Duo and Tan, Zhiquan and Zhao, Linglan and Zhang, Zhongqiang and Fang, Xiangzhong and Huang, Weiran},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025}
}
```