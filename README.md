# A Shift-invariant Neural Network for Bottom-hole Pressure Prediction of Deep-Water Gas Wells

This repo is the Pytorch implementation of LS-Linear: "A Shift-invariant Neural Network for Bottom-hole Pressure Prediction of Deep-Water Gas Wells".

## Features

Beside LS-Linear, we adapt three time series prediction and forecast models for the problem of Bottom-hole Pressure Prediction.
- [x] [DLinear](https://arxiv.org/pdf/2205.13504) (AAAI 2023)
- [x] [TSMixer](https://arxiv.org/pdf/2303.06053) (Transactions on Machine Learning Research 2023)
- [x] [PatchTST](https://arxiv.org/abs/2211.14730) (ICLR 2023)

## Getting Started
### Environment Requirements

dependencies:
  - python=3.10
  - numpy
  - pytorch-gpu
  - cudatoolkit
  - pandas
  - mlflow

### Dataset

Please contact jiangjie_AT_cup.edu.cn to get a copy of the dataset.

### Commandline for running baselines

```python run_exp.py --exp_name=run1 --exp_train --learner_train_epochs=30 --mlflow_log_by_steps=100 --splitted_dataset_ratio=7:1:2 --model_name=linear```

### Commandline for running LS-Linear

```python run_exp.py --exp_name=run1 --exp_train --learner_train_epochs=30 --mlflow_log_by_steps=100 --splitted_dataset_ratio=7:1:2 --model_name=lts --learn_to_scale_model=linear```
