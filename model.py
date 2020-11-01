import numpy as np
import cv2


class model():
    """
    docstring
    """
    def __init__(self, in_out_size, n_layer, layer_size):
        assert type(in_out_size) == type(in_out_size) == tuple
        assert type(n_layer) == int
        self.num_layer = n_layer
        self.in_size = in_out_size[0]
        self.out_size = in_out_size[1]
        self.layer_size = layer_size
        self.initLayers()
        self.initWeights()

    def initLayers(self):
        """
        Initialize input layer, output layer and hidden layers.
        """
        pass
    
    def initWeights(self):
        pass


