import torch
import torch.nn as nn


class PreProcessor:
    '''
    Class to preprocess models before binarizing them
    '''

    def __init__(self):
        pass

    def _load_model_from_file(model_path):
        # TODO: Implement this later on
        pass


    def _load_pretrained_models():
        # TODO: Implement this later on
        pass


    def flatten_model(self, model):
        flattened_model = []

        for mod in model.modules():
            if(list(mod.children()) == []):
                flattened_model.append(mod)
    
        return nn.Sequential(*flattened_model)