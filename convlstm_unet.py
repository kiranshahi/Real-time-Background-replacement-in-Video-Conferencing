import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, Conv2DTranspose, SpatialDropout3D, ConvLSTM2D, TimeDistributed
from keras.layers.core import Activation, Permute
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger


def get_model():
    inputs = Input((256, 256, 256, 3))

    # list of number of filters per block
    depth_cnn = [32, 64, 128, 256]

    # start of encoder block
    # encoder block1
    conv11 = Conv3D(depth_cnn[0], (3, 3, 3), padding='same', name='conv1_1')(inputs)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    conc11 = concatenate([inputs, conv11], axis=4)
    conv12 = Conv3D(depth_cnn[0], (3, 3, 3), padding='same', name='conv1_2')(conc11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)
    conc12 = concatenate([inputs, conv12], axis=4)
    perm = Permute((3, 1, 2, 4))(conc12)
    pool1 = TimeDistributed(MaxPooling2D((2, 2)), name='pool1')(perm)
    pool1 = Permute((2, 3, 1, 4))(pool1)
    pool1 = SpatialDropout3D(0.1)(pool1)

    # encoder block2
    conv21 = Conv3D(depth_cnn[1], (3, 3, 3), padding='same', name='conv2_1')(pool1)
    conv21 = BatchNormalization()(conv21)
    conv21 = Activation('relu')(conv21)
    conc21 = concatenate([pool1, conv21], axis=4)
    conv22 = Conv3D(depth_cnn[1], (3, 3, 3), padding='same', name='conv2_2')(conc21)
    conv22 = BatchNormalization()(conv22)
    conv22 = Activation('relu')(conv22)
    conc22 = concatenate([pool1, conv22], axis=4)
    perm = Permute((3, 1, 2, 4))(conc22)
    pool2 = TimeDistributed(MaxPooling2D((2, 2)), name='pool2')(perm)
    pool2 = Permute((2, 3, 1, 4))(pool2)

    pool2 = SpatialDropout3D(0.1)(pool2)

    # encoder block3
    conv31 = Conv3D(depth_cnn[2], (3, 3, 3), padding='same', name='conv3_1')(pool2)
    conv31 = BatchNormalization()(conv31)
    conv31 = Activation('relu')(conv31)
    conc31 = concatenate([pool2, conv31], axis=4)
    conv32 = Conv3D(depth_cnn[2], (3, 3, 3), padding='same', name='conv3_2')(conc31)
    conv32 = BatchNormalization()(conv32)
    conv32 = Activation('relu')(conv32)
    conc32 = concatenate([pool2, conv32], axis=4)
    perm = Permute((3, 1, 2, 4))(conc32)
    pool3 = TimeDistributed(MaxPooling2D((2, 2)), name='pool3')(perm)

    pool3 = SpatialDropout3D(0.1)(pool3)

    # end of encoder block
    # ConvLSTM block
    x = BatchNormalization()(ConvLSTM2D(filters=depth_cnn[3], kernel_size=(3,3), padding='same', return_sequences=True)(pool3))
    x = BatchNormalization()(ConvLSTM2D(filters=depth_cnn[3], kernel_size=(3,3), padding='same', return_sequences=True)(x))
    x = BatchNormalization()(ConvLSTM2D(filters=depth_cnn[3], kernel_size=(3,3), padding='same', return_sequences=True)(x))

    # start of decoder block
    # decoder block1

    up1 = TimeDistributed(Conv2DTranspose(depth_cnn[2], (2, 2), strides=(2, 2), padding='same', name='up1'))(x)
    up1 = Permute((2, 3, 1, 4))(up1)
    up6 = concatenate([up1, conc32], axis=4)
    conv61 = Conv3D(depth_cnn[2], (3, 3, 3), padding='same', name='conv4_1')(up6)
    conv61 = BatchNormalization()(conv61)
    conv61 = Activation('relu')(conv61)
    conc61 = concatenate([up6, conv61], axis=4)
    conv62 = Conv3D(depth_cnn[2], (3, 3, 3), padding='same', name='conv4_2')(conc61)
    conv62 = BatchNormalization()(conv62)
    conv62 = Activation('relu')(conv62)
    conv62 = concatenate([up6, conv62], axis=4)

    # decoder block2
    up2 = Permute((3, 1, 2, 4))(conv62)
    up2 = TimeDistributed(Conv2DTranspose(depth_cnn[1], (2, 2), strides=(2, 2), padding='same'), name='up2')(up2)
    up2 = Permute((2, 3, 1, 4))(up2)
    up7 = concatenate([up2, conv22], axis=4)
    conv71 = Conv3D(depth_cnn[1], (3, 3, 3), padding='same', name='conv5_1')(up7)
    conv71 = BatchNormalization()(conv71)
    conv71 = Activation('relu')(conv71)
    conc71 = concatenate([up7, conv71], axis=4)
    conv72 = Conv3D(depth_cnn[1], (3, 3, 3), padding='same', name='conv5_2')(conc71)
    conv72 = BatchNormalization()(conv72)
    conv72 = Activation('relu')(conv72)
    conv72 = concatenate([up7, conv72], axis=4)

    # decoder block3
    up3 = Permute((3, 1, 2, 4))(conv72)
    up3 = TimeDistributed(Conv2DTranspose(depth_cnn[0], (2, 2), strides=(2, 2), padding='same', name='up3'))(up3)
    up3 = Permute((2, 3, 1, 4))(up3)
    up8 = concatenate([up3, conv12], axis=4)
    conv81 = Conv3D(depth_cnn[0], (3, 3, 3), padding='same', name='conv6_1')(up8)
    conv81 = BatchNormalization()(conv81)
    conv81 = Activation('relu')(conv81)
    conc81 = concatenate([up8, conv81], axis=4)
    conv82 = Conv3D(depth_cnn[0], (3, 3, 3), padding='same', name='conv6_2')(conc81)
    conv82 = BatchNormalization()(conv82)
    conv82 = Activation('relu')(conv82)
    conc82 = concatenate([up8, conv82], axis=4)

    # end of decoder block
    output = Conv3D(1, (1, 1, 1), activation='sigmoid', name='final')(conc82)

    return Model(inputs=[inputs], outputs=output)

# Hyperparameters
IMAGE_SIZE = 256
EPOCHS = 15
BATCH = 15
LR = 1e-4
model_path = "/home/kiran_shahi/dissertation/model/convlstm_unet.h5"
csv_path = "/home/kiran_shahi/dissertation/log/convlstm_data.csv"

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


ds = pd.read_csv("image.csv")
train_dataset = tf_dataset(ds['image'].tolist(), ds['mask'].tolist(), batch=BATCH)

model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks)