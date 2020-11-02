import numpy as np


class layer:
    """
    Class of neural network layer.
    """
    def __init__(self, _size, prev_size= None, data=None, _type="hidden_output", activation="sigmoid"):
        assert _type in ["input", "hidden_output"]
        self._type = _type
        self._size = _size
        if data is not None:
            self.data = np.reshape(np.array(data), (_size, 1))
        else:
            self.data = np.array((_size, 1))
        if self._type=="hidden_output":
            self.unactivated = np.array((_size, 1))
            self.weights = np.array((_size, prev_size))
        else:
            self.unactivated = self.weights = None
            
    def init(self, _input):
        """
        Initialize layer data and also weights if not input layer.
        """
        assert type(_input) == np.ndarray and _input.shape == self.data.shape
        self.data = _input
        
    def update(self, _input):
        """
        Update unactivated neurons with input data and then activate them.
        """
        assert type(_input) == np.ndarray and _input.shape == self.data.shape
        self.unactivated = _input
        self.activate()
        
        
    
    def activate(self):
        """
        Activate the unactivated neurons and store in data.
        """
        
        
      

class model:
    """
    Class of neural network structure and data included.
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
        self.input_layer = layer(self.in_size, _type="input")
        self.hidden_layers = list()
        for n in self.layer_size:
            list.append(layer(n))
        self.output_layer = layer(self.out_size, self.output_layer)
    
    def initWeights(self):
        """
        Initialize weights between layers.
        """


    


