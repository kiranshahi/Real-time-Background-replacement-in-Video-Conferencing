import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Add, ReLU, Conv2DTranspose, Concatenate, Dropout
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, UpSampling2D, AveragePooling2D, Reshape, ConvLSTM2D
from tensorflow.keras.models import Model


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x / 255.0
    x = x.astype(np.float32)
    # (256, 256, 3)
    return x


def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
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


def conv_block(inputs, num_filters):
    x = Conv2D(filters=num_filters, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def upsample_block(inputs, skip_features, num_filters):
    x = UpSampling2D(interpolation='bilinear')(inputs)
    x = Concatenate()([x, skip_features])
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
    x1 = Activation('relu')(x)
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

    return Model(inputs, outputs)


# Hyperparameters
IMAGE_SIZE = 560
EPOCHS = 100
BATCH = 30
LR = 1e-4
model_path = "mobilev3_unet.h5"
csv_path = "data.csv"

model = get_model()
model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(LR),
    metrics=[
        tf.keras.metrics.MeanIoU(num_classes=2),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.Precision()
    ])

callbacks = [
    ModelCheckpoint(model_path, monitor="loss", verbose=1),
    ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4),
    CSVLogger(csv_path),
    EarlyStopping(monitor='loss', patience=10, restore_best_weights=False)
]



ds = pd.read_csv("image.csv")
train_dataset = tf_dataset(ds['image'].tolist(), ds['mask'].tolist(), batch=BATCH)

model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks)
