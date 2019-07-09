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

x_train, y_train, x_test, y_test = load_mnist(False, True, crop_size, [0,1]) # Load data

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

timings = {}

#
# Variable importance measures
#
rate, rate_time = RATE_BNN(bnn, x_test)
timings["RATE"] = rate_time

# Compute feature importances
start_time = time.time()
pixel_corr = np.array([pearsonr(x_test[:,j], y_test[:,0])[0] for j in range(p)])
timings["Pixel-corr"] = time.time() - start_time

# RF mimic with no CV
rf_mimic_nocv = RandomForestClassifier()
y_train_mimic = most_common_bnn_prediction(bnn, x_train, n_mc_samples)
start_time = time.time()
rf_mimic_nocv.fit(x_train, y_train_mimic)
timings["RF_noCV"] = time.time() - start_time
rf_mimic_imp_gini = rf_mimic_nocv.feature_importances_

# with CV
rf_mimic_withCV_, rf_withCV_time = get_rf_mimic(bnn, x_train, x_test)
timings["RF_withCV"] = rf_withCV_time
            
# GBM mimic with no CV
gbm_mimic_nocv = GradientBoostingClassifier()
start_time = time.time()
gbm_mimic_nocv.fit(x_train, y_train_mimic)
gbm_mimic_imp_gini = gbm_mimic_nocv.feature_importances_
timings["GBM_noCV"] = time.time() - start_time

# with CV
gbm_mimic_withCV_, gbm_withCV_time = get_gbm_mimic(bnn, x_train, x_test)
timings["GBM_withCV"] = gbm_withCV_time

# train simple logistic regression on each pixel using predicted response from BNN (mimic modeling). pull out slope and rank them.
start_time = time.time()
lm_mimic_coeffs, lm_fit_time = get_lm_mimic_coefficients(bnn, x_train, x_test, n_mc_samples=n_mc_samples)
timings["LM"].append(lm_fit_time)

#
# Now shuffle pixels and record test accuracy
#
n_test_samples = 20
n_shuffle_repeats = 10
n_selection_repeats = 5
n_shuffled_pixels = np.arange(0, p+1, 4)

def get_test_accuracy(n_shuffled, imp_vals):
    x_cp = np.copy(x_test)
    for pix_idx in np.where(rank_array(imp_vals)<n_shuffled)[0]:
        x_cp[:,pix_idx] = np.random.permutation(x_cp[:,pix_idx])
    return bnn.score(x_cp, y_test, n_test_samples)


imp_vals = {"RATE_high" : rate, 
            "Pixel-corr_high" : np.abs(pixel_corr), 
            "lm_mimic_high" : np.abs(lm_mimic_coeffs), 
            "rf_gini_high_withCV" : rf_mimic_withCV_.feature_importances_,
            "rf_gini_high_noCV" : rf_mimic_imp_gini, 
            "gbm_gini_high_noCV" : gbm_mimic_imp_gini,
            "gbm_gini_high_withCV" : gbm_mimic_withCV_.feature_importances_}

with open("plot3c_data.pkl", "wb") as f:
    pickle.dump(imp_vals, f)

res = {k : [] for k in imp_vals.keys()}
res["random"] = []

for n_pix in n_shuffled_pixels:
    
    print("Shuffling {} pixels...".format(n_pix))
    
    accuracies = {k :[] for k in res.keys()}
    
    for k in res.keys():
        print(k, end=", ")
        if k != "random":
            for _ in range(n_shuffle_repeats):
                accuracies[k].append(get_test_accuracy(n_pix, imp_vals[k]))
        elif k == "random":
            for _ in range(n_selection_repeats):
                for _ in range(n_shuffle_repeats):
                    accuracies[k].append(get_test_accuracy(n_pix, np.random.choice(p, replace=False, size=n_pix)))
        else:
            print("Invalid key {}".format(k))
    
    for k, v in accuracies.items():
        res[k].append(v)
        
    with open("plotc_data.pkl", "wb") as f:
        pickle.dump(res, f)
        print("saved")
