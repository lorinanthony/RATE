import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import EarlyStopping

from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc

import time, pickle

from rate import *

n_workers = int(sys.argv[1])
print("Using {} workers".format(n_workers))

n_mc_samples = 10

n_vals = [int(val) for val in [1e3, 1e4, 1e5]]
p_vals = [int(val) for val in [1e2, 3e2, 1e3]]
methods = ['Pixel-corr', 'RATE', 'RF-mimic-gini-noCV', 'RF-mimic-gini-withCV', 'GBM-mimic-gini-noCV', 'GBM-mimic-gini-withCV']

# Network architecture
def network_layers(p):
    layers = []
    layers.append(tf.keras.layers.Dense(512, activation='relu', input_shape=(p,)))
    layers.append(tf.keras.layers.Dense(512, activation='relu'))
    layers.append(tfp.layers.DenseLocalReparameterization(1))
    return layers

final_result = {}

for n in n_vals:
    for p in p_vals:
        print(n, p)
        final_result[(n,p)] = {m : [] for m in methods}
        successful_trains = 0
        while successful_trains < 10:
            print("repeat number {}".format(successful_trains))

            p_informative = int(0.1*p)
            p_redundant = 0

            n_clust_per_class = 3

            flipped_label_frac = 0.1

            shift = None
            scale = [1.0] + [1.0 for i in range(p-1)]

            test_size = 0.3

            X, y = make_classification(n_samples=n, n_features=p, n_informative=p_informative,
                                       n_redundant=p_redundant, n_repeated=0, n_classes=2, n_clusters_per_class=n_clust_per_class,
                                       flip_y=flipped_label_frac, shift=shift, scale=scale, shuffle=False)
            y = y[:,np.newaxis]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            bnn = BNN_Classifier(network_layers(p), p, 1)
            fit_history = bnn.fit(X_train, y_train, epochs=25, batch_size=128, validation_split=0.3,
                    callbacks=[EarlyStopping(monitor="val_acc", patience=3)],
                    verbose=0)

            if not np.isfinite([item for sublist in fit_history.history.values() for item in sublist]).all():
                print("Skipping due to nan in loss/accuracy")
                continue
            print("BNN score is ", bnn.score(X_test, y_test, n_mc_samples, True))

            # Compute feature importances
            start_time = time.time()
            final_result[(n,p)]["Pixel-corr"].append([np.array([pearsonr(X_test[:,j], y_test[:,0])[0] for j in range(p)]), time.time()-start_time])
            final_result[(n,p)]["RATE"].append(RATE_BNN(bnn, X_test, n_workers=n_workers))

            # RF mimic with no CV
            rf_mimic_nocv = RandomForestClassifier()
            y_train_mimic = most_common_bnn_prediction(bnn, X_train, n_mc_samples)
            start_time = time.time()
            rf_mimic_nocv.fit(X_train, y_train_mimic)
            final_result[(n,p)]["RF-mimic-gini-noCV"].append([rf_mimic_nocv.feature_importances_, time.time()-start_time])

            # RF mimic with CV
            rf_mimic_out = get_rf_mimic(bnn, X_train, X_test, 10, n_jobs=n_workers)
            final_result[(n,p)]["RF-mimic-gini-withCV"].append([rf_mimic_out[0].feature_importances_, rf_mimic_out[1]])
            
            # GBM mimic with no CV
            gbm_mimic_nocv = GradientBoostingClassifier()
            start_time = time.time()
            gbm_mimic_nocv.fit(X_train, y_train_mimic)
            final_result[(n,p)]["GBM-mimic-gini-noCV"].append([gbm_mimic_nocv.feature_importances_, time.time()-start_time])

            # GBM mimic with CV
            gbm_mimic_out = get_gbm_mimic(bnn, X_train, X_test, 10, n_jobs=n_workers)
            final_result[(n,p)]["GBM-mimic-gini-withCV"].append([gbm_mimic_out[0].feature_importances_, gbm_mimic_out[1]])
            
            with open("simstudy_data.pkl", "wb") as f:
                pickle.dump(final_result, f)

            successful_trains += 1
