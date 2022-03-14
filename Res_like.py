import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.datasets import mnist
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
class CNNBLock(tf.keras.layers.Layer):
    def __init__(self,num_channels,out_channels,kernel_size=3):
        super(CNNBLock,self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=num_channels,kernel_size=kernel_size,padding='same')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels,kernel_size=3,padding='same')
    def call(self,X):
        Y = X
        X = self.conv1(X)
        X = self.batch_norm(X)
        X = self.relu(X+Y)
        X = self.conv2(X)
        return X

class Res_like(tf.keras.Model):
    def __init__(self,cnnblocks,num_classes):
        super(Res_like,self).__init__()
        self.cnnblocks = cnnblocks
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self,X):
        for cnnblock in cnnblocks:
            X=cnnblock(X)
        X=self.pool(X)
        X=self.classifier(X)
        return X

if __name__ ==  "__main__":
    cnnblocks = [
             CNNBLock(32,64),CNNBLock(64,128),CNNBLock(128,256)
]   
    tfmodel = Res_like(cnnblocks=cnnblocks, num_classes=10)
    tfmodel.build(input_shape=(None,28,28,1))
    (X_train,y_train),(X_test,y_test)=mnist.load_data()
    X_train = X_train.reshape(-1,28,28,1).astype('float32')
    X_test = X_test.reshape(-1,28,28,1).astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    tfmodel.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
    tfmodel.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=32,epochs=5)
