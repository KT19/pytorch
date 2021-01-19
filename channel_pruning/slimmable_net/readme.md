# Reimplementation of "Slimmable Neural Networks" (Jiahui Yu et al., ICLR2019) [[arxiv](https://arxiv.org/abs/1812.08928)]

## for running
```python3
python3 train.py
```

### Experimental results
![results](plot_results.png)

### Exepriment details
1. Using VGG16 with batch normalization.
2. Optimizer = SGD, lr = 0.1, momentum=0.9, weight_decay = 5e-4
3. Experiments on CIFAR-10
4. Using cosine annealing

### Quantitative results

| model size | accuracy [%] |
|:-:|:-:|
| x0.25 | 85.87 |
| x0.5 | 89.69 |
| x0.75 | 90.41 |
| x1.0 | 90.48 |
