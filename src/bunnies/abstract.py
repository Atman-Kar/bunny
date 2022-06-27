import torch
import torch.nn as nn
from abc import ABC, abstractclassmethod


class Bunny(ABC):
    '''
    Abstract class for the bunnies
    '''

    @abstractclassmethod
    def binarize(self, parameter):
        '''
        binarize input parameters
        '''
        pass


    def swap_layers(self, model, *args, **kwargs):
        
        list_model = list(model.children())
        for idx, layer in enumerate(list_model):
            try:
                layer.weight.data = self.binarize(layer.weight.data)
            except:
                print(f"Cannot binarize")

            try:
                layer.bias.data = self.binarize(layer.bias.data)
            except:
                print(f"Cannot binarize")

            list_model[idx] = layer.type(torch.int8)

        return nn.Sequential(*list_model)
