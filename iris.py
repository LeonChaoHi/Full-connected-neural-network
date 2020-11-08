import numpy as np
from model import Model
from trainer import Trainer
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def main():
    # load Iris dataset
    iris = load_iris()
    n_samples,n_features=iris.data.shape
    print("Number of sample:",n_samples)    #Number of sample: 150
    print("Number of feature",n_features)   #Number of feature 4
    x = iris.data
    y = np.zeros((n_samples, 3))
    y[list(range(n_samples)), iris.target] = 1
    
    # construct and train model
    model = Model((4, 3), (9, 10, 10))
    trainer = Trainer(model, x, y, 0.01, 1000)
    loss_list = trainer.fit()
    plt.plot(loss_list)
    plt.show()
    return
    

    

if __name__ == "__main__":
    main()