import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Normalization
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 10000)
np.set_printoptions(suppress=True)
np_config.enable_numpy_behavior()

def pre_process(file):
    df = pd.read_csv(file)
    df = df.dropna()

    # extract text specific columns
    df.Date = pd.Categorical(df.Date)
    df['Date_code'] = df.Date.cat.codes
    print("...Dataset:\n",df)

    # split dataframe into train and test
    train_df = df.sample(frac=0.5, random_state=0)
    test_df = df.drop(train_df.index)

    # choose attributes(columns to train on)
    X_train = train_df.iloc[:, [2,4,5,6,7,8]]
    Y_train = train_df['Close']
    X_test = test_df.iloc[:, [2,4,5,6,7,8]]
    Y_test = test_df['Close']
    print("...Original X train data:\n", X_train)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test).reshape(-1,1)
    Y_train = np.array(Y_train).reshape(-1,1)
    #print("Numpy array Y:\n", np.array(Y_test, dtype='f'))

    # normalize with numpy by axis
    x_train, x_train_norms = normalize(X_train, axis=0, return_norm=True)# normalize specific columns
    y_train = normalize(Y_train, axis=0)# normalize array(its only 1 column attribute)
    x_test,x_test_norms = normalize(X_test, axis=0, return_norm=True)# normalize columns
    y_test, y_test_norms = normalize(Y_test, axis=0, return_norm=True)# normalize array(its only 1 column attribute)
    print("...Normalized train data\n", x_train)

    # make tensor objects
    X_train = tf.convert_to_tensor(x_train)
    Y_train = tf.convert_to_tensor(y_train)
    X_test = tf.convert_to_tensor(x_test)
    Y_test = tf.convert_to_tensor(y_test)

    return X_train,Y_train, X_test,Y_test,y_test_norms, x_test_norms


def create_dataset(xs, ys):
    trainortestdataset = tf.data.Dataset.from_tensor_slices((xs, ys))
    return trainortestdataset


file = 'archive 2/sp500_stocks.csv'
print("Pre-processing file...")
X_train,Y_train, X_test, Y_test, y_test_norms, x_test_norms = pre_process(file)
#print("creating datasets...")

# Create architecture
model = Sequential()
#norm = model.add(Normalization())
model.add(Dense(units=6, activation='relu', name='layer2'))
model.add(Dense(units=12, activation='relu', name='layer3'))
#model.add(Dense(units=30, activation='relu', name='layer4'))
ouput = model.add(Dense(units=1, activation='sigmoid', name='output'))

# Fit model to Data
print("fitting model...")
model.compile(optimizer='adam',
              loss=tf.losses.MeanSquaredError(),
              #metrics=['accuracy']
              )

history = model.fit(
    X_train,Y_train,
    epochs=3,
    #steps_per_epoch=2,
    validation_data=(X_test, Y_test),
    #validation_steps=2
)

# Evaluate on test dataset and print results
#print("\nEvaluation")
#results = model.evaluate(X_test, Y_test, batch_size=128)
#print("test loss, test acc:", results)

# make predictions
print("predicting...")
predictions = model.predict((X_test, Y_test))
print("...Rescale factor for outputs:",y_test_norms)
predictions = np.multiply(predictions, y_test_norms)
Y_test = np.multiply(Y_test, y_test_norms)
#print("Actual employee count:\n", Y_test)
#print("Predicted employee count:\n", predictions)
d = [np.array(Y_test), np.array(predictions)]
d = np.concatenate((d[0], d[1]),axis=1)
#print("d:", d)
output_df = pd.DataFrame(data=d, columns=['Correct values', 'Predictions'])
print(output_df)
