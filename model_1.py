import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Add, ReLU
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, UpSampling2D, AveragePooling2D
from tensorflow.keras.models import Model


def UpsamplingBlock(x, nfilters):
    x = Conv2D(kernel_size=1, filters=nfilters, padding='same', use_bias=False)(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5)(x)
    x = Activation("relu")(x)
    return x


def BottleneckBlock(channels, inputs):
    x, r = inputs
    a, b = tf.split(x, 2, -1)
    b, r = ConvGRU(channels // 2, [b, r])
    x = tf.concat([a, b], -1)
    return x, r

def LRASPP(x, out_channels=128):
    x1 = Conv2D(out_channels, 1, use_bias=False)(x)
    x1 = BatchNormalization(momentum=0.1, epsilon=1e-5)(x1)
    x1 = ReLU()(x1)

    x2 = Conv2D(out_channels, 1, use_bias=False, activation='sigmoid')(tf.reduce_mean(x, axis=[1, 2], keepdims=True))
    return x1 * x2


def OutputBlock(channels, inputs):
    x, s = inputs
    x = UpSampling2D(interpolation='bilinear')(x)
    # x = tf.image.crop_to_bounding_box(x, 0, 0, tf.shape(s)[1], tf.shape(s)[2])

    x = tf.concat([x, s], -1)

    x = Conv2D(channels, 3, padding='SAME', use_bias=False)(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5)(x)
    x = ReLU()(x)
    x = Conv2D(channels, 3, padding='SAME', use_bias=False)(x)
    x = BatchNormalization(momentum=0.1, epsilon=1e-5)(x)
    x = ReLU()(x)

    return x

encoder = MobileNetV3Large(input_shape=(256, 256, 3), weights="imagenet", include_top=False)

x = encoder.layers[193].output
x = LRASPP(x)

# s1 = AveragePooling2D(padding="SAME")(encoder.layers[38].output)
# s2 = AveragePooling2D(padding="SAME")(encoder.layers[34].output)
# s3 = AveragePooling2D(padding="SAME")(encoder.layers[16].output)



x = UpsamplingBlock(x, 72)
x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
x = Add()([x, encoder.layers[38].output])


x = UpsamplingBlock(x, 72)
x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
x = Add()([x, encoder.layers[34].output])


x = UpsamplingBlock(x, 64)
x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
x = Add()([x, encoder.layers[16].output])


x = UpsamplingBlock(x, 16)
x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)

#x = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)

x = OutputBlock(16, x)

model = Model(encoder.input, x)

#x = Reshape((1, x.shape[1], x.shape[2], x.shape[3]))(x)
#x = ConvLSTM2D(16, kernel_size=3, padding='same', return_sequences=False, activation='relu')(x)

IMAGE_SIZE = 256
EPOCHS = 100
BATCH = 30
LR = 1e-4
model_path = "test_mdl/model.h5"
csv_path = "data.csv"

opt = tf.keras.optimizers.Nadam(LR)
metrics = ['mse']
model.compile(loss="mse", optimizer=opt, metrics=metrics)

dataset_path = "/home/kiran_shahi/dissertation/dataset/new_data"
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "test")

callbacks = [
    ModelCheckpoint(model_path, monitor="val_loss", verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
    CSVLogger(csv_path),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
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


#valid_dataset = tf_dataset(valid_x, valid_y, batch=BATCH)

#train_steps = len(train_x) // BATCH
#valid_steps = len(valid_x) // BATCH


# if len(train_x) % BATCH != 0:
#     train_steps += 1
# if len(valid_x) % BATCH != 0:
#     valid_steps += 1


model.fit(
    train_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# model.fit(
#     train_dataset,
#     validation_data=valid_dataset,
#     epochs=EPOCHS,
#     steps_per_epoch=train_steps,
#     validation_steps=valid_steps,
#     callbacks=callbacks
# )