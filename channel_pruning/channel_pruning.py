#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn as nn
import torchvision

"""
Implementation of Pruning filters for efficient convnets
arxiv:https://arxiv.org/abs/1608.08710
"""

class PruneFilter():
    def __init__(self):
        pass

    def prune_layer(self, target, next_layer, pruned_ratio, batchnorm=None, criteria="l1",last=False):
        """
        args:
        target: target convolution layer
        next_layer: next convolution layer
        pruned_ratio: ratio of pruned filter
        batchnorm(optional): batch normalization layer
        last(optional): if last convolution, true

        return:
        target: pruned target convolution layer
        next_layer: prunned next convolution layer
        batchnorm: batchnorm layer
        """

        #number of remained filters
        remained_num = target.out_channels - int(pruned_ratio*target.out_channels)

        #step1
        kernel_weight_dict = {}
        for i in range(target.out_channels):
            kernel_weight = torch.sum(torch.abs(target.weight[i]))
            kernel_weight_dict[i] = kernel_weight

        #step2
        kernel_weight_dict = sorted(kernel_weight_dict.items(), key=lambda x: x[1])

        #step3
        remained_list = []
        if criteria == "random":
            remained_list = [i for i in range(target.out_channels)]
            remained_list = np.random.choice(remained_list, remained_num)
        else:
            for (key,value) in kernel_weight_dict:
                remained_list.append(key)
            remained_list = remained_list[target.out_channels-remained_num:]

        #step4
        target.weight = nn.Parameter(target.weight[remained_list])
        if target.bias is not None:
            target.bias = nn.Parameter(target.bias[remained_list])
        target.out_channels = remained_num

        if not last:
            next_layer.in_channels = remained_num
            next_layer.weight = nn.Parameter(next_layer.weight[:,remained_list])
        else:
            next_layer.in_features = remained_num
            next_layer.weight = nn.Parameter(next_layer.weight[remained_list])
            if next_layer.bias is not None:
                next_layer.bias = nn.Parameter(next_layer.bias[remained_list])

        if batchnorm is not None:
            batchnorm.weight = nn.Parameter(batchnorm.weight[remained_list])
            if batchnorm.bias is not None:
                batchnorm.bias = nn.Parameter(batchnorm.bias[remained_list])
            batchnorm.running_var = batchnorm.running_var[remained_list]
            batchnorm.running_mean = batchnorm.running_mean[remained_list]

        return target, next_layer, batchnorm

    def prune(self, model, act, criteria="l1"):
        """
        args:
        model: target model
        act: information of how layer is pruned :list [layer_num, ratio]
        criteria(optional): criterion of pruning,
        if criteria is 'random', the filter is randomly pruned

        return:
        pruned model
        """

        #number of target layer's filter
        layer_num = int(act[0])
        ratio = float(act[1])

        last = False
        conv_num = 0
        conv_dict = {} #for indices to conv layer num
        batch_dict = {} #for batch normalization

        """
        check target index
        """
        for k,layer in enumerate(model.features):
            if isinstance(layer,nn.Conv2d):
                conv_num = conv_num + 1
                conv_dict[conv_num] = k
            if isinstance(layer,nn.BatchNorm2d):
                batch_dict[conv_num] = k #i.e. corresponding to conv


        target_layer = model.features[conv_dict[layer_num]]
        if len(batch_dict) > 0:
            batchnorm = model.features[batch_dict[layer_num]]

        if layer_num + 1 in conv_dict:
            next_layer = model.features[conv_dict[layer_num+1]]
        else: #i.e. last conv layer
            next_layer = model.classifier[0]
            last = True

        target_layer, next_layer, batchnorm = self.prune_layer(target_layer, next_layer,ratio,criteria=criteria,last=last)


        return model

def test():
    method = PruneFilter()
    model = torchvision.models.vgg16(pretrained=True)
    print("before pruning")
    print(model)

    x = torch.randn(1, 3, 224,244)
    act = [[12,0.8],
    [1,0.9],
    [4,0.3],
    ]
    for a in act:
        method.prune(model, a)
    print("after pruning")
    print(model)
    x = model(x) #confirm


if __name__ == "__main__":
    test()
