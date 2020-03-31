# Re-Implementation of The Lottery Ticket Hypothesis at Scale
This repository is the re-Implementation of the LT at Scale.
Using late initialization to prune the cnn model

## Experimental Setup(in this repo)
1. dataset・・・cifar10
2. epochs・・・30
3. cnn model・・・tiny version (please check train.py and modules.py)

![The results of above setting](results.png)

### How to use?(example)
First, train model
***
python3 train.py
***
The results are automatically saved in log_file in csv format

For visualizing plot,
***
python3 plot.py
***
