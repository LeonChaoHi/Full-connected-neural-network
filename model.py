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
        self.prev_size = prev_size
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
            self.weights = np.random.rand(self._size, self.prev_size) - 0.5
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
        self.data = 1 / (1 + np.exp(-self.unactivated))
        
      

class Model:
    """
    Class of neural network structure and data included.
    """
    def __init__(self, in_out_size, hidden_layer_size):
        assert type(in_out_size) == type(hidden_layer_size) == tuple
        self.in_size = in_out_size[0]
        self.out_size = in_out_size[1]
        self.hidden_size = hidden_layer_size
        self.n_hidden_layer = len(hidden_layer_size)       # number of hidden layers
        self.initLayers()

    def initLayers(self):
        """
        Initialize input layer, output layer and hidden layers.
        """
        self.input_layer = NNLayer(self.in_size, _type="input")
        self.hidden_layers = list()
        for n in range(self.n_hidden_layer):
            prev_size = self.in_size if n==0 else self.hidden_size[n-1]
            self.hidden_layers.append(NNLayer(self.hidden_size[n], prev_size, _type="hidden"))
        self.output_layer = NNLayer(self.out_size, self.hidden_size[-1], _type="output")
    
    def feedForward(self, _input):
        """Feedforward by input to every layer of model.
        Args:
            input ([np.ndarray]): [Input data]
        """
        self.input_layer.updateData(_input)
        for i in range(self.n_hidden_layer):
            current_layer = self.hidden_layers[i]
            prev_layer = self.input_layer if i == 0 else self.hidden_layers[i-1]
            new_data = current_layer.weights.dot(prev_layer.data) + current_layer.intercept
            current_layer.updateData(new_data)
        new_data = self.output_layer.weights.dot(self.hidden_layers[-1].data) + self.output_layer.intercept
        self.output_layer.updateData(new_data)
            
        
    def backProp(self, lr, y):
        """Backpropagation
        Args:
            lr (float): learning rate
            loss (float): loss computed from the loss function
        """
        assert y.shape == self.output_layer.data.shape
        
        out = self.output_layer
        hs = self.hidden_layers
        _in = self.input_layer
        
        def derivative_sigmoid(x):
            return x * (1 - x)
        
        # update output layer's parameters
        out.dz = derivative_sigmoid(out.data) * (out.data - y)
        o_dw = out.dz.dot(hs[-1].data.T)
        o_db = out.dz
        out.updateWeights(out.weights-lr*o_dw, out.intercept-lr*o_db)
        
        # update hidden layers' parameters, from last to begin
        for i in range(self.n_hidden_layer-1, -1, -1):
            prev_h =  _in if i==0 else hs[i-1]
            h = hs[i]
            next_h = out if i==self.n_hidden_layer-1 else hs[i+1]
            
            h.dz = derivative_sigmoid(h.data) * next_h.weights.T.dot(next_h.dz)
            h_dw = h.dz.dot(prev_h.data.T)
            h_db = h.dz
            h.updateWeights(h.weights-lr*h_dw, h.intercept-lr*h_db)
        
        return
        