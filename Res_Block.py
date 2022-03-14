import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

class ResidualBlock(tf.keras.layers.Layer):
  def __init__(self,num_channels,out_channels,strides,is_used_conv11=False,*kwargs):
    super(ResidualBlock, self).__init__(*kwargs)
    self.is_used_conv11=is_used_conv11
    self.conv1=tf.keras.layers.Conv2D(num_channels,kernel_size=(3,3),padding='same',strides=1)
    self.batch_norm=tf.keras.layers.BatchNormalization()
    self.relu=tf.keras.layers.ReLU()
    #2nd convolution layer
    self.conv2=tf.keras.layers.Conv2D(num_channels,kernel_size=3,strides=1,padding='same')
    if self.is_used_conv11:
      self.conv3=tf.keras.layers.Conv2D(num_channels,kernel_size=1,strides=1,padding='same')
    #last convolution layer
    self.conv4=tf.keras.layers.Conv2D(out_channels,kernel_size=1,strides=1,padding='same')
    self.relu=tf.keras.layers.ReLU()


  def call(self,X):
    if self.is_used_conv11:
      Y=self.conv3(X)
    else:
      Y=X
    X=self.conv1(X)
    X=self.batch_norm(X)
    X=self.relu(X)
    X=self.conv2(X)
    X=self.batch_norm(X)
    X=self.relu(X+Y)
    X=self.conv4(X)
    X=self.relu(X)
    return X


class ResNet18(tf.keras.Model):
  def __init__(self,residual_blocks,output_shape):
    super(ResNet18,self).__init__()
    self.conv1=tf.keras.layers.Conv2D(filters=64,kernel_size=7,strides=2,padding='same')
    self.batch_norm=tf.keras.layers.BatchNormalization()
    self.relu=tf.keras.layers.ReLU()
    self.residual_blocks=residual_blocks
    self.max_pool=tf.keras.layers.MaxPool2D(pool_size=3,strides=2)
    self.global_avg_pool=tf.keras.layers.GlobalAveragePooling2D()
    self.dense=tf.keras.layers.Dense(units=output_shape)

  def call(self,X):
    X=self.conv1(X)
    X=self.batch_norm(X)
    X=self.relu(X)
    X=self.max_pool(X)
    for residual_block in residual_blocks:
      X=residual_block(X)
    X=self.global_avg_pool(X)
    X=self.dense(X)
    return X

if __name__ == "__main__":
    residual_blocks = [
    # Two start conv mapping
    ResidualBlock(num_channels=64, out_channels=64, strides=2, is_used_conv11=False),
    ResidualBlock(num_channels=64, out_channels=64, strides=2, is_used_conv11=False),
    # Next three [conv mapping + identity mapping]
    ResidualBlock(num_channels=64, out_channels=128, strides=2, is_used_conv11=True),
    ResidualBlock(num_channels=128, out_channels=128, strides=2, is_used_conv11=False),
    ResidualBlock(num_channels=128, out_channels=256, strides=2, is_used_conv11=True),
    ResidualBlock(num_channels=256, out_channels=256, strides=2, is_used_conv11=False),
    ResidualBlock(num_channels=256, out_channels=512, strides=2, is_used_conv11=True),
    ResidualBlock(num_channels=512, out_channels=512, strides=2, is_used_conv11=False)
]

model = ResNet18(residual_blocks, output_shape=10)
model.build(input_shape=(None, 28, 28, 1))
model.summary()
