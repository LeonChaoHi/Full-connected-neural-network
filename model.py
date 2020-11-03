import numpy as np
import cv2


class NNLayer:
    """
    Class of neural network layer.
    """
    def __init__(self, _size, prev_size= None, _type="hidden"):
        assert _type in ["input", "hidden", "output"]
        self._type = _type
        self._size = _size
        self.prevsize = prev_size
        self.initData()

    def initData(self):
        """
        Initialize layer data.
        """
        self.data = np.zeros((self._size, 1))
        if self._type == "input":
            self.unactivated = self.weights = self.intercept = None
        else:
            self.unactivated = np.zeros_like(self.data)
            self.weights = np.random.rand(self._size, self.prev_size)
            self.intercept = np.zeros_like(self.data)

    def updateData(self, _input):
        """
        Update unactivated neurons with input data and then activate them.
        """
        assert _input.shape == self.data.shape
        if self._type == "input":
            self.data = _input
        else:
            self.unactivated = _input
            self.activate()
        
    def updateWeights(self, new_weights, new_intercept):
        """Update value of weights.

        Args:
            _inputs ([np.ndarray]): [new weights]
        """
        assert new_weights.shape == self.weights.shape
        self.weights = new_weights
        self.intercept = new_intercept
    
    def activate(self):
        """
        Activate the unactivated neurons and store in data.
        """
        self.data = 1 / (1 + np.exp(self.unactivated))
        
      

class Model:
    """
    Class of neural network structure and data included.
    """
    def __init__(self, in_out_size, n_hidden_layer, layer_size):
        assert type(in_out_size) == type(in_out_size) == tuple
        assert type(n_hidden_layer) == int
        self.num_layer = n_hidden_layer
        self.in_size = in_out_size[0]
        self.out_size = in_out_size[1]
        self.layer_size = layer_size
        self.initLayers()

    def initLayers(self):
        """
        Initialize input layer, output layer and hidden layers.
        """
        self.input_layer = NNLayer(self.in_size, _type="input")
        self.hidden_layers = list()
        for n in self.layer_size:
            list.append(NNLayer(n, _type="hidden_output"))
        self.output_layer = NNLayer(self.out_size, self.layer_size[-1], _type="hidden_output")
    
    def feedForward(self, _input):
        """Feedforward by input to every layer of model.
        Args:
            input ([np.ndarray]): [Input data]
        """
        self.input_layer.updateData(_input)
        for i in range(self.n_hidden_layer):
            current_layer = self.hidden_layers[i]
            prev_layer = self.input_layer if i == 0 else self.hidden_layers[i-1]
            new_data = prev_layer.data.dot(current_layer.weights) + current_layer.intercept
            current_layer.updateData[new_data]
        new_data = self.hidden_layers[-1].data.dot(self.output_layer.weights) + self.output_layer.intercept
        self.output_layer.updateData(new_data)
            
        
    def backProp(self, lr, loss):
        """Backpropagation
        Args:
            lr (float): learning rate
            loss (float): loss computed from the loss function
        """
        

