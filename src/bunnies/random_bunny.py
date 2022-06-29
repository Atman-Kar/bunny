import torch

from .abstract import Bunny


class RandomBunny(Bunny):
    def __init__(self):
        super().__init__()

    def binarize(self, parameter):
        return torch.randint(low=0, high=15, size=parameter.size())
