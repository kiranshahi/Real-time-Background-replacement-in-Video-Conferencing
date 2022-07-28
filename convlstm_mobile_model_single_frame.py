import os
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Dropout, concatenate
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, UpSampling2D, Reshape, ConvLSTM2D
from tensorflow.keras.models import Model


def conv_block(inputs, num_filters):
    x = ConvLSTM2D(filters=num_filters, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def upsample_block(inputs, skip_features, num_filters):
    x = UpSampling2D(interpolation='bilinear')(inputs)
    N = x.shape[1]
    x1 = Reshape(target_shape=(1, np.int32(N), np.int32(N), num_filters))(x)
    x2 = Reshape(target_shape=(1, np.int32(N), np.int32(N), num_filters))(skip_features)

    x = concatenate([x1, x2], axis=1)
    x = conv_block(x, num_filters)
    x = Dropout(0.3)(x)
    return x


def output_block(inputs):
    x = Conv2D(3, (1, 1), 1, 'same', use_bias=False)(inputs)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(3, (1, 1), 1, 'same', use_bias=False)(inputs)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    return x


def LRASPP(x, out_channels: int):
    x1 = Conv2D(out_channels, 1, use_bias=False)(x)
    x1 = BatchNormalization(momentum=0.1, epsilon=1e-5)(x1)
    x1 = Activation('relu')(x1)
    x2 = Conv2D(out_channels, 1, use_bias=False, activation='sigmoid')(tf.reduce_mean(x, axis=[1, 2], keepdims=True))
    return x1 * x2


def get_model():
    inputs = Input(shape=(560, 560, 3), name='input')
    encoder = MobileNetV3Large(input_tensor=inputs, weights="imagenet", include_top=False)

    for layer in encoder.layers:
        layer.trainable = False

    # Encoder
    e1 = encoder.get_layer('input').output  # [(None, 256, 256, 3)
    e2 = encoder.get_layer('re_lu_2').output  # (None, 128, 128, 64)
    e3 = encoder.get_layer('re_lu_6').output  # (None, 64, 64, 72)
    e4 = encoder.get_layer('re_lu_15').output  # (None, 32, 32, 240)

    # Bridge
    b1 = encoder.get_layer('re_lu_29').output  # (None, 16, 16, 672)

    lraspp = LRASPP(b1, 672)

    # Decoder
    d1 = upsample_block(lraspp, e4, 512)
    d2 = upsample_block(d1, e3, 256)
    d3 = upsample_block(d2, e2, 128)
    d4 = upsample_block(d3, e1, 64)

    # Output
    outputs = output_block(d4)
    outputs = Conv2D(1, (1, 1), 1, 'same', activation='sigmoid')(outputs)
    model = Model(inputs, outputs)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(LR),
        metrics=[
            tf.keras.metrics.MeanIoU(num_classes=2),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision()
        ])

    return model


# Hyperparameters
IMAGE_SIZE = 560
EPOCHS = 150
BATCH = 8
LR = 1e-4


# model = get_model()
def get_callback():
    model_path = "mobilev3_Convlstm.h5"
    csv_path = "mobilev3_Convlstm.csv"
    callbacks = [
        ModelCheckpoint(model_path, monitor="loss", verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4),
        CSVLogger(csv_path),
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=False)
    ]
    return callbacks


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x / 255.0
    x = x.astype(np.float32)
    return x


def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x


def preprocess(image_path, mask_path):
    def f(img_pth, msk_pth):
        img_pth = img_pth.decode()
        msk_pth = msk_pth.decode()
        x = read_image(img_pth)
        y = read_mask(msk_pth)
        return x, y

    image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    mask.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])
    return image, mask


def tf_dataset(images, masks, batch=30):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset


def load_data():
    train_path = '/home/kiran_shahi/dissertation/old_dataset/new_data/train'
    test_path = '/home/kiran_shahi/dissertation/old_dataset/new_data/test'

    train_x = sorted(glob(os.path.join(train_path, "image/*")))
    train_y = sorted(glob(os.path.join(train_path, "mask/*")))

    test_x = sorted(glob(os.path.join(test_path, "image/*")))
    test_y = sorted(glob(os.path.join(test_path, "mask/*")))
    return (train_x, train_y), (test_x, test_y)


def fit_model():
    (train_x, train_y), (test_x, test_y) = load_data()
    train_dataset = tf_dataset(train_x, train_y, batch=4)
    valid_dataset = tf_dataset(test_x, test_y, batch=4)
    model.fit(train_dataset, validation_data=valid_dataset, epochs=EPOCHS, callbacks=callbacks, batch_size=BATCH)
