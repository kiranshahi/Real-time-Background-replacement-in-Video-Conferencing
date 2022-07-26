import os
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, UpSampling2D, DepthwiseConv2D, Add
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.models import Model

IMAGE_SIZE = 256
EPOCHS = 100
BATCH = 8
LR = 1e-4
smooth = 1e-15


def bottleneck(x, n_filters):
    y = DepthwiseConv2D(kernel_size=3, depth_multiplier=1, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    y = Conv2D(kernel_size=1, filters=n_filters, padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    y = DepthwiseConv2D(kernel_size=3, depth_multiplier=1, padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    y = Conv2D(kernel_size=1, filters=n_filters, padding='same')(y)
    y = BatchNormalization()(y)

    z = Conv2D(kernel_size=1, filters=n_filters, padding='same')(x)
    z = BatchNormalization()(z)

    z = Add()([y, z])
    z = Activation("relu")(z)

    return z


def build_model():
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")
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

    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    return Model(inputs, x)


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


def load_data(path):
    images = sorted(glob(os.path.join(path, "image", "*png")))
    masks = sorted(glob(os.path.join(path, "mask", "*png")))

    train_x, valid_x = train_test_split(images, test_size=0.2, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=0.2, random_state=42)
    return (train_x, train_y), (valid_x, valid_y)


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x / 255.0
    x = x.astype(np.float32)
    return x


def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
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


def tf_dataset(images, masks, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset


def set_data():
    dataset_path = "/home/kiran_shahi/dissertation/old_dataset/new_data"
    train_path = os.path.join(dataset_path, "train")

    (train_x, train_y), (valid_x, valid_y) = load_data(train_path)
    train_x, train_y = shuffling(train_x, train_y)

    train_dataset = tf_dataset(train_x, train_y, batch=BATCH)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=BATCH)

    train_steps = len(train_x) // BATCH
    valid_steps = len(valid_x) // BATCH

    if len(train_x) % BATCH != 0:
        train_steps += 1
    if len(valid_x) % BATCH != 0:
        valid_steps += 1
    return (train_dataset, valid_dataset), (train_steps, valid_steps)


def call_model():
    model = build_model()
    opt = tf.keras.optimizers.Nadam(LR)
    metrics = [dice_coef, Recall(), Precision()]
    model.compile(loss=dice_loss, optimizer=opt, metrics=metrics)

    csv_path = "/home/kiran_shahi/dissertation/log/mobilenetv3_unet_single_frame.csv"
    model_path = '/home/kiran_shahi/dissertation/model/mobilenetv3_unet_single_frame.h5'
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
        EarlyStopping(monitor='val_loss', patience=10),
        CSVLogger(csv_path),
        ModelCheckpoint(model_path, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
    ]

    (train_dataset, valid_dataset), (train_steps, valid_steps) = set_data()

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )


call_model()
