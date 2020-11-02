import numpy as np
import model


class trainer:
    def __init__(self, model, input_data, output_data, learning_rate, batch_size, steps, epochs):
      self.model = model
      self.input = input_data
      self.output = output_data
      self.lr = learning_rate
      self.batch_size = batch_size
      self.steps = steps
      self.epochs = epochs
      
    def feedForward(self, input):
        """Feedforward by input to every layer of model.

        Args:
            input ([np.ndarray]): [Input data]
        """
        model = self.model
        model = model.model()
        model.input_layer.data = 
        
        
        
    def backProp(self, parameter_list):
        """
        docstring
        """
        pass