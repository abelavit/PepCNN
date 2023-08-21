# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:18:50 2023

@author: abelac
"""

import numpy as np
import pandas as pd
import warnings
import pickle
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import tensorflow as tf
import tensorflow.keras.layers as tfl


# load data that excludes the test data
file = open("dataset1_Train_Positives_rerun.dat",'rb')
positive_set = pickle.load(file)
file = open("dataset1_Train_Negatives_All_rerun.dat",'rb')
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
file = open("dataset1_Test_Samples_rerun.dat",'rb')
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
    model.add(tfl.Dropout(0.23))
    model.add(tfl.Conv1D(128, 3, padding='same',activation='relu'))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(0.21))
    model.add(tfl.Conv1D(64, 3, padding='same',activation='relu'))
    model.add(tfl.BatchNormalization())
    model.add(tfl.Dropout(0.47))
  
    model.add(tfl.Flatten())
    
    #model.add(tfl.Dense(1000, activation='relu'))
    model.add(tfl.Dense(128, activation='relu'))
    #model.add(tfl.Dropout(0.2))
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

# Train the Model
batch_size = 30
epochs = 200

checkpoint = tf.keras.callbacks.ModelCheckpoint("dataset1_best_model_weights_rerun.h5", save_best_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=3, restore_best_weights=True)
history = cnn_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[checkpoint, early_stopping])

df_loss_auc = pd.DataFrame(history.history)
df_loss= df_loss_auc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_auc= df_loss_auc[['auc','val_auc']]
df_auc.rename(columns={'auc':'train','val_auc':'validation'},inplace=True)
Model_Loss_plot_title = 'Model Loss'
df_loss.plot(title=Model_Loss_plot_title,figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
Model_AUC_plot_title = 'Model AUC'
df_auc.plot(title=Model_AUC_plot_title,grid=True,figsize=(12,8)).set(xlabel='Epoch',ylabel='AUC')

eval_result = cnn_model.evaluate(X_test, Y_test)
print(f"test loss: {round(eval_result[0],4)}, test auc: {round(eval_result[1],4)}")
