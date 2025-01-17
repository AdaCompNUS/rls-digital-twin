# Load Variables from Existing Checkpoints
import os
import argparse
import pprint
import numpy as np
import copy
import torch

from nose.tools import *


class LayerSpec(object):
    """The details of a particular layer"""

    def __init__(self, layerType, name, weight=None, bias=None):
        self.type = layerType
        self.name = name
        self.weight = weight
        self.bias = bias
        self.output_spikes = {}
        self.membrane = None
        self.spikes = None
        self.mem_aggregate = None  # use to verify output match that of pre-trained DNN


class ParseModel(object):
    """Parse the model parameters from the checkpoint"""

    def __init__(self, layer_list, model):
        self.layer_list = layer_list
        self.layers = []
        self.model = model

    def parseLayers(self):
        self.layers = []
        for layer in self.layer_list:
            if layer.startswith('conv'):
                weight = self.model.state_dict()[layer + '.weight']
                bias = self.model.state_dict()[layer + '.bias']

                bnBeta = self.model.state_dict()[layer + '_bn.bias']
                bnGamma = self.model.state_dict()[layer + '_bn.weight']
                bnMean = self.model.state_dict()[layer + '_bn.running_mean']
                bnVar = self.model.state_dict()[layer + '_bn.running_var']

                # re-parameterization by integrating the beta and gamma factors
                # into the 'conv' layer weights
                ratio = torch.div(bnGamma, torch.sqrt(bnVar))
                weightNorm = torch.transpose(torch.mul(torch.transpose(weight, 0, 3), ratio), 0, 3)
                biasNorm = torch.mul((bias - bnMean), ratio) + bnBeta

                assert_equal(weight.size(), weightNorm.size())
                assert_equal(bias.size(), biasNorm.size())

                newLayer = LayerSpec('conv', layer, weightNorm, biasNorm)
                self.layers.append(newLayer)

            elif layer.startswith('linear'):
                weight = self.model.state_dict()[layer + '.weight']
                bias = self.model.state_dict()[layer + '.bias']

                newLayer = LayerSpec('linear', layer, torch.transpose(weight, 0, 1), bias)
                self.layers.append(newLayer)

            else:
                print("{} layer is not understood.".format(layer))
        return self.layers
