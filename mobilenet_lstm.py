import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, UpSampling2D, DepthwiseConv2D, Add, \
    Reshape, ConvLSTM2D
from tensorflow.keras.models import Model

from data_image import get_data
from metrics import dice_coef, dice_loss, mad

IMAGE_SIZE = 256
EPOCHS = 100
BATCH = 8
LR = 1e-4


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


def lstm_layer(x, n_filters):
    row_col_sze = x.shape[1]
    x = Reshape(target_shape=(1, np.int32(row_col_sze), np.int32(row_col_sze), n_filters))(x)
    x = ConvLSTM2D(filters=n_filters, kernel_size=(3, 3), padding='same', return_sequences=False,
                   kernel_initializer='he_normal')(x)
    return x


def build_model():
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")
    encoder = MobileNetV3Large(input_tensor=inputs, weights="imagenet", include_top=False)
    x = encoder.layers[193].output

    x = bottleneck(x, 72)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Add()([x, encoder.layers[38].output])
    x = lstm_layer(x, 72)

    x = bottleneck(x, 72)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Add()([x, encoder.layers[34].output])
    x = lstm_layer(x, 72)

    x = bottleneck(x, 64)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Add()([x, encoder.layers[16].output])
    x = lstm_layer(x, 64)

    x = bottleneck(x, 16)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = lstm_layer(x, 16)

    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    return Model(inputs, x)

def call_model():
    model = build_model()
    model.compile(
        loss=dice_loss,
        optimizer=tf.keras.optimizers.Adam(LR),
        metrics=[
            dice_coef,
            mad,
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.MeanIoU(num_classes=2)
        ])

    csv_path = "/home/kiran_shahi/dissertation/log/mobilenetv3_lstm.csv"
    model_path = '/home/kiran_shahi/dissertation/model/mobilenetv3_lstm.h5'
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10),
        CSVLogger(csv_path),
        ModelCheckpoint(model_path, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
    ]

    (train_dataset, valid_dataset), (train_steps, valid_steps) = get_data(BATCH, False)

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )

call_model()
