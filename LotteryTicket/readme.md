# The Lottery Ticket Hypothesis
## This repository is an implementation of LT

## In this repo, there are two settings
1. On MNIST, using simple MLP (mlp_mnist)
2. On CIFAR10, using CNN (cnn_cifar)

# How to use?
Default settings are written in configure.py
You can change learning rate, pruned_ratio (i.e., how many parameters are pruned in total), and iteration (gradually prune parameters).

## Example
Please enter the following command in each file.
***
python3 train.py --epochs 10 --lr 1.2e-3 --iteration 10 --ratio 0.95
***

Results are automatically saved in csv format and png (for ploting).
