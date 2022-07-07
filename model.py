import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Add, ReLU, Conv2DTranspose, Concatenate
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, UpSampling2D, AveragePooling2D, Reshape, ConvLSTM2D
from tensorflow.keras.models import Model


def conv_block(input, num_filters):
    x = Conv2D(filters=num_filters, kernel_size=3, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def upsample_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def get_model():
    inputs = Input(shape=(256, 256, 3), name='input')
    encoder = MobileNetV3Large(input_tensor=inputs, weights="imagenet", include_top=False)

    # Encoder block
    e1 = encoder.get_layer('input').output  # [(None, 256, 256, 3)
    e2 = encoder.get_layer('re_lu_2').output  # (None, 128, 128, 64)
    e3 = encoder.get_layer('re_lu_6').output  # (None, 64, 64, 72)
    e4 = encoder.get_layer('re_lu_15').output  # (None, 32, 32, 240)

    # Bridge
    b1 = encoder.get_layer('re_lu_29').output  # (None, 16, 16, 672)

    # Decoder
    d1 = upsample_block(b1, e4, 512)
    d2 = upsample_block(d1, e3, 256)
    d3 = upsample_block(d2, e2, 128)
    d4 = upsample_block(d3, e1, 64)

    outputs = Conv2D(3, (1, 1), 1, 'same', activation='softmax')(d4)

    return Model(inputs, outputs)

# Hyperparameters
IMAGE_SIZE = 256
EPOCHS = 100
BATCH = 30
LR = 1e-4
model_path = "mobilev3_unet.h5"
csv_path = "data.csv"

model = get_model()
model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(LR), metrics=['mse'])

callbacks = [
    ModelCheckpoint(model_path, monitor="val_mse", verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
    CSVLogger(csv_path)
]


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    x = x.astype(np.float32)
    # (256, 256, 3)
    return x


def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
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


def tf_dataset(images, masks, batch=30):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset

ds = pd.read_csv("image.csv")
train_dataset = tf_dataset(ds['image'].tolist(), ds['mask'].tolist(), batch=BATCH)

model.fit( train_dataset, epochs=EPOCHS, callbacks=callbacks)