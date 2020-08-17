from src.models import Bike_Classifier
from src.run_model import run_model
from data.load_data import load_proj_data
from data.my_dataset import MyDataset

import matplotlib.pyplot as plt
import numpy as np
import csv
import os

print("started")
# train_count = number of training samples
# test_count = number of testing samples
trainX, testX, trainY, testY = load_proj_data(train_count, test_count)
train_set = MyDataset(trainX, trainY)
test_set = MyDataset(testX, testY)
model = Bike_Classifier()
model, loss, acc = run_model(model,'train',train_set,batch_size=50,learning_rate=1e-4,n_epochs=10)
test_loss, test_accuracy = run_model(model,'test',test_set=test_set,batch_size=50)

print(acc['train'])
epochs = len(loss['train'])
plt.figure()
plt.title('Training Loss by Epoch')
plt.plot(np.arange(epochs), loss['train'])
plt.show()

plt.figure()
plt.title('Training Accuracy by Epoch')
plt.plot(np.arange(epochs), acc['train'])
plt.show()

print("test loss: ", test_loss, "test accuracy: ", test_accuracy)
