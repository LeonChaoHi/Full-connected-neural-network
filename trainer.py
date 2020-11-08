import numpy as np
from model import Model


class Trainer:
    def __init__(self, model, input_data, output_data, learning_rate, epochs):
        self.model = model
        self._input = input_data
        self.output = output_data
        self.lr = learning_rate
        # self.batch_size = batch_size
        self.epochs = epochs
        # self.epochs = epochs
      
    def fit(self):
        model = self.model
        # model = Model()
        datasize = self._input.shape[0]
        total_loss = list()
        for epoch in range(self.epochs):
            epoch_loss = list()
            epoch_accuracy = list()
            for i in range(datasize):
                model.feedForward(self._input[i].reshape(model.in_size, 1))
                epoch_loss.append(0.5 * ((model.output_layer.data - self.output[i].reshape(model.out_size, 1))**2).sum())
                epoch_accuracy.append(int(model.output_layer.data.argmax()==self.output[i].argmax()))
                model.backProp(self.lr, self.output[i].reshape(model.out_size, 1))
            epoch_loss = sum(epoch_loss) / datasize
            epoch_accuracy = sum(epoch_accuracy) / datasize
            total_loss.append(epoch_loss)
            if epoch%100 == 0:
                print("Epoch: ", epoch, "  Loss:", epoch_loss, " Accuracy:", epoch_accuracy)
        return total_loss