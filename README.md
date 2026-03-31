# DyKAF: Dynamical Kronecker Approximation of the Fisher Information Matrix for Gradient Preconditioning

## New CIFAR 10 time experiments

![New CIFAR 10 time experiments](figures/cifar_test_f1_vs_relative_time.png)



Main optimizer: `./src/optimizers/dykaf.py`

Syntetic experiments: `./src/libsvm/`

Fine-tuning experiments: `./src/fine_tuning/`

Pre-training: implement DyKAF from `./src/optimizers/dykaf.py` into pretrain optimization [LIB](https://github.com/epfml/llm-baselines/tree/soap)