import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Activation, Conv2DTranspose, Input, Conv3D, concatenate, TimeDistributed, Dropout

from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

seq_size = 15


def conv_block(inputs, num_filters):
    x = ConvLSTM2D(filters=num_filters, kernel_size=(3, 3), padding='same', return_sequences=True, kernel_initializer='he_normal', recurrent_dropout=0.3, dropout=0.3)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)
    return x


def decoder_block(inputs, skip_features, num_filters):
    x = TimeDistributed(Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same"))(inputs)
    x = concatenate([x, skip_features], axis=-1)
    x = conv_block(x, num_filters)
    return x


def resNet_UNET():
    temporal_input = Input(shape=(None, 256, 256, 3), name="temporal_input")
    inputs = Input((256, 256, 3))
    encoder = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    for layer in encoder.layers:
        layer.trainable = False

    e2 = TimeDistributed(Model(encoder.input, encoder.get_layer("conv1_relu").output), name='time_distributed_model_1')(temporal_input)
    e3 = TimeDistributed(Model(encoder.input, encoder.get_layer("conv2_block3_out").output), name='time_distributed_model_2')(temporal_input)
    e4 = TimeDistributed(Model(encoder.input, encoder.get_layer("conv3_block4_out").output), name='time_distributed_model_3')(temporal_input)

    # Bridge
    b1 = TimeDistributed(Model(encoder.input, encoder.get_layer("conv4_block6_out").output), name='time_distributed_model_4')(temporal_input)  ## (32 x 32 x 1024)

    # Decoder
    d1 = decoder_block(b1, e4, 512)  ## (64 x 64)
    d2 = decoder_block(d1, e3, 256)  ## (128 x 128)
    d3 = decoder_block(d2, e2, 64)  ## (256 x 256)

    outputs = TimeDistributed(Conv2DTranspose(15, (2, 2), strides=2, padding="same"))(d3)
    outputs = Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same")(outputs)

    return Model(inputs=temporal_input, outputs=outputs)


# Hyperparameters
IMAGE_SIZE = 256
EPOCHS = 100
BATCH = 2
LR = 1e-4
model_path = "/home/kiran_shahi/dissertation/model/resnet_unet_convlstm_aug_val.h5"
csv_path = "/home/kiran_shahi/dissertation/log/resnet_data_aug_convlstm_val.csv"

model = resNet_UNET()
model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(LR),
    metrics=[
        tf.keras.metrics.MeanIoU(num_classes=2),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.Precision()
    ])

callbacks = [
    ModelCheckpoint(model_path, monitor="loss", verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4),
    CSVLogger(csv_path),
    EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
]


def read_image(images_path):
    images = []
    for path in images_path:
        path = path.decode()
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
        x = x / 255.0
        x = x.astype(np.float32)
        images.append(x)
    return np.array(images)


def read_mask(masks_path):
    masks = []
    for path in masks_path:
        path = path.decode()
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
        x = x / 255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=-1)
        masks.append(x)
    return np.array(masks)


def preprocess(image_path, mask_path):
    def f(image_path, mask_path):
        x = read_image(image_path)
        y = read_mask(mask_path)
        return x, y

    image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([seq_size, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask.set_shape([seq_size, IMAGE_SIZE, IMAGE_SIZE, 1])
    return image, mask


def tf_dataset(images, masks, batch=4):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch, drop_remainder=True)
    dataset = dataset.prefetch(1)
    return dataset


def image_seq(images):
    img_list = []
    sub_list = []
    count = 0
    for image in images:
        count = count + 1
        if count != seq_size:
            sub_list.append(image)
        else:
            sub_list.append(image)
            img_list.append(sub_list)
            sub_list = []
            count = 0
    return img_list


ds = pd.read_csv("image.csv")

train_dataset = tf_dataset(image_seq(ds['image'].tolist()), image_seq(ds['mask'].tolist()))

model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks)
