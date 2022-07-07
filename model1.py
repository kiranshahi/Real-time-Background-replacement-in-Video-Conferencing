import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, UpSampling2D, Input
from tensorflow.keras.layers import Concatenate, DepthwiseConv2D, Add, GRU, ConvLSTM2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV2

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K

from tensorflow.keras.optimizers import SGD, Adam


def bottleneck(x, nfilters):
    y = DepthwiseConv2D(kernel_size=3, depth_multiplier=1, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    y = Conv2D(kernel_size=1, filters=nfilters, padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    y = DepthwiseConv2D(kernel_size=3, depth_multiplier=1, padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    y = Conv2D(kernel_size=1, filters=nfilters, padding='same')(y)
    y = BatchNormalization()(y)

    z = Conv2D(kernel_size=1, filters=nfilters, padding='same')(x)
    z = BatchNormalization()(z)

    z = Add()([y, z])
    z = Activation("relu")(z)

    return z

def model():
    inputs = Input(shape=(256, 256, 3), name="input_image")

    encoder = MobileNetV3Large(input_tensor=inputs, weights="imagenet", include_top=False)
    x = encoder.layers[193].output

    x = bottleneck(x, 72)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Add()([x, encoder.layers[38].output])

    x = bottleneck(x, 72)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Add()([x, encoder.layers[34].output])

    x = bottleneck(x, 64)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Add()([x, encoder.layers[16].output])

    x = bottleneck(x, 16)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)

    x = Reshape((1, x.shape[1], x.shape[2], x.shape[3]))(x)
    x = ConvLSTM2D(16, kernel_size=3, padding='same', return_sequences=False, activation='relu')(x)

    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    return Model(inputs, x)

IMAGE_SIZE = 256
EPOCHS = 50
BATCH = 8
LR = 1e-4

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(dataset_path):
    x = sorted(glob(os.path.join(dataset_path, "image", "*png")))
    y = sorted(glob(os.path.join(dataset_path, "mask", "*png")))
    return x, y

model = model()
opt = tf.keras.optimizers.Nadam(LR)
metrics = [dice_coef, Recall(), Precision()]
model.compile(loss=dice_loss, optimizer=opt, metrics=metrics)

dataset_path = "D:\\Learn\\Image Processing\\Human Image Segmentation DeepLabv3\\new_data"
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "test")

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
]

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    x = x.astype(np.float32)
    # (256, 256, 3)
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def preprocess(image_path, mask_path):
    def f(image_path, mask_path):
        image_path = image_path.decode()
        mask_path = mask_path.decode()
        x = read_image(image_path)
        y = read_mask(mask_path)
        return x, y
    image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([256, 256, 3])
    mask.set_shape([256, 256, 1])
    return image, mask

train_x, train_y = load_data(train_path)
train_x, train_y = shuffling(train_x, train_y)
valid_x, valid_y = load_data(valid_path)

def tf_dataset(images, masks, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset

train_dataset = tf_dataset(train_x, train_y, batch=BATCH)
valid_dataset = tf_dataset(valid_x, valid_y, batch=BATCH)

train_steps = len(train_x)//BATCH
valid_steps = len(valid_x)//BATCH

if len(train_x) % BATCH != 0:
    train_steps += 1
if len(valid_x) % BATCH != 0:
    valid_steps += 1

model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    callbacks=callbacks
)
