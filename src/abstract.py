from abc import ABC, abstractclassmethod


class Bunny(ABC):
    '''
    Abstract class for the bunnies
    '''

    @abstractclassmethod
    def binarize(self):
        '''
        binarize input model 
        '''
        pass
