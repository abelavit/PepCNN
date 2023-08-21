# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:18:50 2023

@author: abelac
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score 
import warnings
import pickle
import copy
from matplotlib import pyplot
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import tensorflow as tf
import tensorflow.keras.layers as tfl

# load data that excludes the test data
file = open("dataset2_Train_Positives.dat",'rb')
positive_set = pickle.load(file)
file = open("dataset2_Train_Negatives_All.dat",'rb')
negative_set_entire = pickle.load(file)
column_names = ['Code','Protein_len','Seq_num','Amino_Acid','Position','Label','Peptide','Mirrored','Feature','Prot_positives']
# randomly pick negative samples to balance it with positve samples (1.5x positive samples)
Negative_Samples = negative_set_entire.sample(n=round(len(positive_set)*1.5), random_state=42)

# combine positive and negative sets to make the final dataset
Train_set = pd.concat([positive_set, Negative_Samples], ignore_index=True, axis=0)

# collect the features and labels of train set
np.set_printoptions(suppress=True)
X_val = [0]*len(Train_set)
for i in range(len(Train_set)):
    feat = Train_set['Feature'][i]
    X_val[i] = feat
X_train_orig = np.asarray(X_val)
y_val = Train_set['Label'].to_numpy(dtype=float)
Y_train_orig = y_val.reshape(y_val.size,1)

# Generate a random order of elements with np.random.permutation and simply index into the arrays Feature and label 
idx = np.random.permutation(len(X_train_orig))
X_train,Y_train = X_train_orig[idx], Y_train_orig[idx]
scaler = StandardScaler()
scaler.fit(X_train) # fit on training set only
X_train = scaler.transform(X_train) # apply transform to the training set

# load test data
file = open("dataset2_Test_Samples.dat",'rb')
Independent_test_set = pickle.load(file)
# collect the features and labels for independent set
X_independent = [0]*len(Independent_test_set)
for i in range(len(Independent_test_set)):
    feat = Independent_test_set['Feature'][i]
    X_independent[i] = feat
X_test = np.asarray(X_independent)
y_independent = Independent_test_set['Label'].to_numpy(dtype=float)
Y_test = y_independent.reshape(y_independent.size,1)
X_test = scaler.transform(X_test) # apply standardization (transform) to the test set

def CNN_Model():
    
    model = tf.keras.Sequential()
    model.add(tfl.Conv1D(128, 5, padding='same', activation='relu', input_shape=(feat_shape,1)))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(0.38))
    model.add(tfl.Conv1D(128, 3, padding='same',activation='relu'))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(0.38))
    model.add(tfl.Conv1D(64, 3, padding='same',activation='relu'))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(0.38))
  
    model.add(tfl.Flatten())
    
    model.add(tfl.Dense(128, activation='relu'))
    model.add(tfl.Dense(32, activation='relu'))
    model.add(tfl.Dense(1, activation='sigmoid'))
    
    return model

feat_shape = X_train[0].size

cnn_model = CNN_Model()

learning_rate = 0.000001
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
cnn_model.compile(optimizer=optimizer,
                   loss='binary_crossentropy',
                   metrics=['AUC'])

cnn_model.summary()

# load the trained weights
cnn_model.load_weights('dataset2_best_model_weights.h5')

eval_result = cnn_model.evaluate(X_test, Y_test)

print(f"test loss: {round(eval_result[0],4)}, test auc: {round(eval_result[1],4)}")
Inde_test_prob = cnn_model.predict(X_test)


def round_based_on_thres(probs_to_round, set_thres):
    for i in range(len(probs_to_round)):
        if probs_to_round[i] <= set_thres:
            probs_to_round[i] = 0
        else:
            probs_to_round[i] = 1
    return probs_to_round

# calculate the metrics
set_thres = 0.885
copy_Probs_inde = copy.copy(Inde_test_prob)
round_based_on_thres(copy_Probs_inde, set_thres)
fpr, tpr, thresholds = roc_curve(Y_test, Inde_test_prob)
inde_auc = round(roc_auc_score(Y_test, Inde_test_prob),4)
inde_pre = round(precision_score(Y_test, copy_Probs_inde),4)
cm = confusion_matrix(Y_test, copy_Probs_inde) # for acc, sen, and spe calculation
total_preds=sum(sum(cm))
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
TP = cm[1,1]
inde_sen = round(TP/(TP+FN),4)
inde_spe = round(TN/(TN+FP),4)

# display the metrics
print(f'Independent Sen: {inde_sen}')
print(f'Independent Spe: {inde_spe}')
print(f'Independent Pre: {inde_pre}')
print(f'Independent AUC: {inde_auc}')

# plot ROC curve
legend = 'AUC = ' + str(inde_auc)
pyplot.figure(figsize=(12,8))
pyplot.plot([0,1], [0,1], linestyle='--')
pyplot.plot(fpr, tpr, marker='.', label=legend)
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()
