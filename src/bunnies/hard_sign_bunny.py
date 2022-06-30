import torch

from .abstract import Bunny


class HardSignBunny(Bunny):
    """
    Apply a hard signum function on the parameters of the model.

                ____
                |
                |     1 ; if (x > 0)
    Sign(x) =   |     0 ; if (x == 0)
                |    -1 ; if (x < 0)
                |___
    """

    def __init__(self):
        super().__init__()

    def binarize(self, parameter):
        """
        Apply the signum function to input parameters

        """
        return torch.sgn(parameter)
