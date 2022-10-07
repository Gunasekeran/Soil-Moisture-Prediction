#pip install tensorflow
#pip install keras
#pip install numpy
#pip install pandas

import tensorflow as tf
import numpy as np
import pandas as pd
import keras.backend as K

#Prediction of Moisture 3 using Moisture 0, 1 and 2
input_columns = ['year', 'month', 'day', 'hour', 'minute', 'moisture0', 'moisture1', 'moisture2']
prediction = ['moisture3']

#splitting the dataset as training and testing sets
train_size = 4000
test_size = 1000

#Defining the function for returning the sum of absolute values
def sum_error(y_true, y_pred):
        return K.sum(K.abs(y_pred - y_true))

#Evaluation Function calling
K.eval(K.sum(K.abs(np.array([0,3]) - np.array([2,5]))))

#Reading the Dataset
data_frame = pd.read_csv('soil_moisture_dataset.csv')
data_frame
data_frame = data_frame.sample(frac=1)

#Dropping a Column, i.e. 'Condition'
data_frame.drop(columns=['condition'])

#Plotting the AxesSubplot of moisture 0, 1, 2 and 3
data_frame[['moisture0', 'moisture1','moisture2', 'moisture3']].plot()

#Assigning the values for Training and Testing Purpose
X_train = data_frame[input_columns][:train_size]
Y_train = data_frame[prediction][:train_size]
X_test = data_frame[input_columns][train_size:]
Y_test = data_frame[prediction][train_size:]

#Displaying the number of rows and columns of the DataFrame
X_train.shape, Y_train.shape

#Using sequential model to display output shape and parameters
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=len(input_columns), input_shape=(len(input_columns),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(units=len(prediction), input_shape=(len(input_columns),)))
model.summary()

#Compiling the model to finalise the model and making it completely ready to use
model.compile(optimizer='adam', loss=sum_error, metrics=['MSE'])

#Fitting training data to measure how the model generalizes to similar data
model.fit(X_train,Y_train, epochs=10)

#Evaluating to check whether the model is best fit for the given problem and corresponding data
model.evaluate(X_test, Y_test)

#Setting weights in Keras using numpy array
weights = np.array(model.get_weights())

#Using get_weights() to return the weights of the model as a list of Numpy arrays
model.get_weights()

#convert dataframe into numpy array
Y_test.to_numpy()[58]

#Using predict() method for the actual prediction (to generate output predictions for the input samples)
prediction = model.predict(X_test.to_numpy())

#Displaying the accuracy value of moisture 3
count = 0
for i in range(len(X_test)):
    if np.abs(prediction[i][0] - Y_test.to_numpy()[i]) > 0.1:
        count += 1
print(count/len(X_test))




#Prediction of Moisture 4 using Moisture 0, 1 and 2
input_columns = ['year', 'month', 'day', 'hour', 'minute', 'moisture0', 'moisture1', 'moisture2']
prediction = ['moisture4']

#splitting the dataset as training and testing sets
train_size = 4000
test_size = 1000

#Defining the function for returning the sum of absolute values
def sum_error(y_true, y_pred):
        return K.sum(K.abs(y_pred - y_true))

#Evaluation Function calling
K.eval(K.sum(K.abs(np.array([0,3]) - np.array([2,5]))))

#Reading the Dataset
data_frame = pd.read_csv('soil_moisture_dataset.csv')
data_frame
data_frame = data_frame.sample(frac=1)

#Dropping a Column, i.e. 'Condition'
data_frame.drop(columns=['condition'])

#Plotting the AxesSubplot of moisture 0, 1, 2 and 4
data_frame[['moisture0', 'moisture1','moisture2', 'moisture4']].plot()

#Assigning the values for Training and Testing Purpose
X_train = data_frame[input_columns][:train_size]
Y_train = data_frame[prediction][:train_size]
X_test = data_frame[input_columns][train_size:]
Y_test = data_frame[prediction][train_size:]

#Displaying the number of rows and columns of the DataFrame
X_train.shape, Y_train.shape

#Using sequential model to display output shape and parameters
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=len(input_columns), input_shape=(len(input_columns),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(units=len(prediction), input_shape=(len(input_columns),)))
model.summary()

#Compiling the model to finalise the model and making it completely ready to use
model.compile(optimizer='adam', loss=sum_error, metrics=['MSE'])

#Fitting training data to measure how the model generalizes to similar data
model.fit(X_train,Y_train, epochs=10)

#Evaluating to check whether the model is best fit for the given problem and corresponding data
model.evaluate(X_test, Y_test)

#Setting weights in Keras using numpy array
weights = np.array(model.get_weights())

#Using get_weights() to return the weights of the model as a list of Numpy arrays
model.get_weights()

#convert dataframe into numpy array
Y_test.to_numpy()[58]

#Using predict() method for the actual prediction (to generate output predictions for the input samples)
prediction = model.predict(X_test.to_numpy())

#Displaying the accuracy value of moisture 4
count = 0
for i in range(len(X_test)):
    if np.abs(prediction[i][0] - Y_test.to_numpy()[i]) > 0.1:
        count += 1
print(count/len(X_test))