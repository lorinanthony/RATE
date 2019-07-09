import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as K

import pickle, time

from tensorflow.keras.callbacks import EarlyStopping

from rate import *

#
# Get data for binary classification, train BNN and then compute RATE
#

# Settings
crop_size = 5

# Training settings
n_epochs = 20
batch_size = 256

# Test settings - the number of posterior samples
n_mc_samples = 100

rate_values = []
M_B_signs = []

for classes in [(0,1). (1,8)]:

    x_train, y_train, x_test, y_test = load_mnist(False, True, crop_size, classes) # Load data

    p = x_train.shape[1]
    C = y_train.shape[1]
    image_size = int(p**0.5)

    # Network architecture
    layers = []
    layers.append(tf.keras.layers.Reshape([image_size, image_size, 1], input_shape=(p,)))
    layers.append(tf.keras.layers.Conv2D(32, (5, 5), activation='relu'))
    layers.append(tf.layers.Flatten())
    layers.append(tf.keras.layers.Dense(512, activation='relu'))
    layers.append(tfp.layers.DenseLocalReparameterization(C))

    bnn = BNN_Classifier(layers, p, C,) # Create and train network
    fit_history = bnn.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs,
                            validation_split=0.2,
                            callbacks=[EarlyStopping(monitor="val_acc", patience=2)],
                            verbose=0)
        
    print(bnn.score(x_test, y_test, n_mc_samples, True))

    #
    # Variable importance measures
    #
    rate, rate_time, M_B, _ = RATE_BNN(bnn, x_test, return_esa_posterior=True)
    rate_values.append(rate)
    M_B_signs.append(np.sign(M_B))

with open("plot_ab_data.pkl", "wb"):
    pickle.dump(dict(zip(['RATE_values', 'M_B_signs'], [rate_values, M_B_signs])), f)

