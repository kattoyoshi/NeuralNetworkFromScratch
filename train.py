# coding utf-8
import os
import numpy as np
from sklearn.model_selection import train_test_split
import utils
import neuralnet


# load dataset
dataset = utils.mnist_dataset("data")
X_train, X_test, y_train, y_test = dataset.load()

# create validation data
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

# train model 
##### parameters #####
epochs = 1
learning_rate = 0.1
batch_size = 32

patience_n_epoch = 5
######################

# define the network structure
model = neuralnet.MLP_MNIST(input_nodes=784, h1_nodes=128, h2_nodes=128, output_nodes=10, learning_rate=learning_rate)

# batch per epoch
if len(y_train) % batch_size == 0:
    batch_per_epoch = len(y_train) // batch_size
else:
    batch_per_epoch = len(y_train) // batch_size + 1

best_valid_loss = np.inf
cnt = 0

for epoch in range(epochs):
    batch_generator = utils.batch_generator(X_train, y_train, batch_size=batch_size, shuffle=True)
    train_loss_epoch = np.array([])
    for ii in range(batch_per_epoch):
        X_batch, y_batch = next(batch_generator)
        train_loss_batch = model.train(X_batch, y_batch)
        train_loss_epoch = np.append(train_loss_epoch, train_loss_batch)
        # monitor training loss
        if ii % 100 == 0:
            print("Epoch: {}/{}".format(epoch+1, epochs),
                  "n_batch: {0}".format(ii),
                  "Training loss in last batch: {:.4f}".format(train_loss_batch))
    avg_train_loss = np.average(train_loss_epoch)
    valid_loss = model.evaluate(X_valid, y_valid)
    print("Epoch: {}/{} finished".format(epoch+1, epochs), 
          "avg_train_loss: {:.4f}".format(avg_train_loss), 
          "valid_loss: {2:.4f}".format(valid_loss))
    if valid_loss < best_valid_loss:
        print ("--- Validation loss improved from {0:.4f} to {1:.4f} ---".format(best_valid_loss, valid_loss))
        best_valid_loss = valid_loss
    else:
        print ("Validation loss didn't improved")
        cnt += 1
        if cnt >= patience_n_epoch:
            print ("--- Stop training ---")

model.save_model("saved_data")

