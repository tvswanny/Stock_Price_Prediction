#!/usr/bin/env python3

# ID: stock_v6b.py
# CSCI 7850 Final Project due 11/22/2022
# AUTHOR: Todd V. Swanson
# INSTALLATION: MIDDLE TENNESSEE STATE UNIVERSITY
# Convolution + LSTM network to
# predict stock prices

# import modules
# import yfinance as yf
import tensorflow.keras as keras
import numpy as np
from sklearn import preprocessing

# suppress tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# set all parameters and hyperparameters
learning_rate = 0.001
batch_size = 64 
dropout = 0.3
epochs = 1000
test_samples = 500          # save last 250 samples (1 year's data) for test data 
train_samples = 3000        # train and validation samples
val_split = 0.2
samples = test_samples + train_samples  # total samples from data set to be used
features = 5                 # data columns representing features
non_features = 0             # data columns to keep data organized
data_step_width = features + non_features  # total # columns in data set
split_point = int(train_samples * (1-val_split))
steps = 32                   # time steps to evaluate
maxConv = 8                  # longest convolution length to evaluate
sample_length = steps + maxConv - 1  # number of time steps needed for each sample     
first_step_idx = -samples - steps - maxConv + 1  # index of 1st datapoint
last_step_idx = -samples   # index representing sample length
print(first_step_idx, last_step_idx)

### Uncomment this section to get the data
# df = yf.download('SPY') # loads all data to 'data' 
# raw_data = np.array(df)
# raw_data.shape

### Uncomment this section to write data to text file
# # data loaded to SPY.txt for further use
# outfile = open('SPY.txt', 'w')
# for row in range(raw_data.shape[0]):
#     for col in range(raw_data.shape[1]):
#         outfile.write(str(raw_data[row,col])+' ')
#     outfile.write('\n')
# outfile.close()

raw_data = np.loadtxt('SPY.txt')
print('raw_data.shape:', raw_data.shape)

# scale all price data to Adjusted Closing Price data (& removes ACP col)
scaled_data = raw_data.copy()
scale_factor = scaled_data[:,4] / scaled_data[:,3]
print(scale_factor.shape)
for i in range(3):
    scaled_data[:,i] = scaled_data[:,i] * scale_factor
scaled_data = np.delete(scaled_data,[3],axis=1)

# function to normalize data
def normalize(x):              #includes mean & std for pct change
    y_mean = np.mean(x[:,3])
    y_std = np.std(x[:,3])
    x_norm = preprocessing.StandardScaler().fit_transform(x)
    y_value = np.array([x_norm[-1,3], y_mean, y_std])    # returns ACP_norm, y_mean, & y_std to reconstruct ACP
    return x_norm, y_value

# prepare X and Y data samples
X = np.zeros(shape=(samples+1,sample_length,data_step_width))
Y = np.zeros(shape=(samples+1,3))
for i in range(samples+1):
    idxFirst = i + first_step_idx
    idxLast = i + last_step_idx
    X[i,:],Y[i,:] = normalize(scaled_data[idxFirst:idxLast or None])
Y = Y[1:]
X = X[:-1]
print(X.shape, Y.shape)

# divide data into train & test samples
X_train = X[:train_samples]
X_test = X[-test_samples:]
Y_train = Y[:train_samples]
Y_test = Y[-test_samples:]
print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)

# construct model
x1 = y1 = keras.layers.Input(shape=(steps,features))
x2 = y2 = keras.layers.Input(shape=X.shape[1:])
x = [x1,x2]

y1 = keras.layers.Conv1D(32,kernel_size=(1),
                        activation='gelu')(x1)
y2 = keras.layers.Conv1D(32,kernel_size=(maxConv),
                        activation='gelu')(x2)
y = keras.layers.Concatenate(axis=2)([y1,y2])   

y = keras.layers.LSTM(128, activation='tanh',
                      dropout = dropout,
                      kernel_regularizer=keras.regularizers.L1L2(l1=0.1,l2=0.1),
                      return_sequences=True,
                      return_state=False)(y)
y = keras.layers.LSTM(64, activation='tanh',
                      dropout = dropout,
                      kernel_regularizer=keras.regularizers.L1L2(l1=0.1,l2=0.1),
                      return_sequences=False,
                      return_state=False)(y)

y = keras.layers.Dropout(dropout)(y)
y = keras.layers.Dense(128,activation='tanh')(y)
y = keras.layers.Dropout(dropout)(y)
y = keras.layers.Dense(64,activation='tanh')(y)
y = keras.layers.Dense(1)(y)

model = keras.Model(x,y)
model.summary()

model.compile(loss=keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.Adam(learning_rate = learning_rate),
              metrics=keras.metrics.MeanAbsoluteError())

# shuffle data
shuffle = np.random.permutation(Y_train.shape[0])
X2_shuffle = X_train[shuffle,:]
X1_shuffle = X2_shuffle[:,-steps:]
Y_shuffle = Y_train[shuffle,:]
print(X1_shuffle.shape, X2_shuffle.shape, Y_shuffle.shape)

history = model.fit([X1_shuffle,X2_shuffle], Y_shuffle[:,0],
                    validation_split=val_split,
                    callbacks=keras.callbacks.EarlyStopping(monitor='mean_absolute_error',patience=100,
                        verbose=1,restore_best_weights=True),
                    epochs=epochs,
                    verbose=1)

# print validation error every 10 epochs
print('mean_absolute_error:', *['%.5f'%(x) for x in
                                history.history['mean_absolute_error'][0::10]])
print('val_mean_absolute_error:', *['%.5f'%(x) for x in
                                    history.history['val_mean_absolute_error'][0::10]])
print('\nMinimum val_mean_absolute_error=', min(history.history['val_mean_absolute_error']),
     'at Epoch',
      history.history['val_mean_absolute_error'].index(min(history.history['val_mean_absolute_error']))+1)

# save model to indicated name
model.save('stock_prediction.h5',save_format='h5')
