import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')
a=pd.read_csv("mental-state.csv")
print(a)

le=LabelEncoder()
##a['frame.time'] = le.fit_transform(a['frame.time'])
##a['ip.dst'] = le.fit_transform(a['ip.dst'])
##a['ip.src'] = le.fit_transform(a['ip.src'])
a['Label'] = le.fit_transform(a['Label'])
############################ features  ####################################
X=a.drop(['Label'],axis=1)

print(X)
############################   labels  ######################################
Y=a['Label']
print(Y)

############################# traing and testing part #######################
x_train,x_test,y_train,y_test = train_test_split(X,Y,shuffle=True,test_size=0.25, random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import Conv1D,MaxPooling1D ,MaxPooling2D, Dropout, GlobalAveragePooling2D, Activation
from keras.layers import Flatten, Dense
##from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.callbacks import ModelCheckpoint, History
# from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score


##Convert class vectors to binary class matrices. This uses 1 hot encoding.
from tensorflow.keras.utils import to_categorical
y_train_binary = to_categorical(y_train)
y_test_binary = to_categorical(y_test)
print(y_test_binary)
##
########CONVOLUTIONAL NEURAL NETWORK(CNN)######################
epochs = 1

batch_size = 100

##model = tf.keras.Sequential()
IB = Sequential()
print(x_train.shape[1],1)
                                
IB.add(Conv1D(16, 3, padding='same', activation='relu', input_shape=(988,1)))#x.shape[1:])) # Input shape: (96, 96, 1)
IB.add(MaxPooling1D(pool_size=1))
print(1)
                              
IB.add(Conv1D(32, 3, padding='same', activation='relu'))
IB.add(MaxPooling1D(pool_size=1))
IB.add(Dropout(0.25))
print(1)                             
IB.add(Conv1D(64, 3, padding='same', activation='relu'))
IB.add(MaxPooling1D(pool_size=1))
print(1)                              
IB.add(Conv1D(128, 3, padding='same', activation='relu'))
IB.add(MaxPooling1D(pool_size=1))
print(1)                                
IB.add(Conv1D(256, 3, padding='same', activation='relu'))
IB.add(MaxPooling1D(pool_size=1))
print(1)                               
# Convert all values to 1D array
IB.add(Flatten())
print(1)                              
IB.add(Dense(512, activation='relu'))
IB.add(Dropout(0.2))
print(1)
IB.add(Dense(3))

hist = History()
##
####checkpointer = ModelCheckpoint(filepath='checkpoint1.hdf5', verbose=1, save_best_only=True)
##
### Complie Model
IB.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history=IB.fit(x_train, y_train_binary, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test_binary))
IB.summary()
#model.save('model')
##predicted =(model.predict(x_test) > 0.5).astype("int32")
y_pred=np.argmax(IB.predict(x_test), axis=-1)
print("Accuracy",accuracy_score(y_test,y_pred))

##filename = 'pickle.pkl'
##pickle.dump(IB, open(filename, 'wb'))
