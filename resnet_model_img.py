import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Activation, Conv2DTranspose, Input, Conv2D, Reshape, \
    concatenate, Bidirectional

from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger


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
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x =
    return x


def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    row_col_sze = x.shape[1]
    x1 = Reshape(target_shape=(1, np.int32(row_col_sze), np.int32(row_col_sze), num_filters))(x)
    x2 = Reshape(target_shape=(1, np.int32(row_col_sze), np.int32(row_col_sze), num_filters))(skip_features)
    x = concatenate([x1, x2], axis=1)

    x = Bidirectional(ConvLSTM2D(filters=num_filters, kernel_size=(3, 3), padding='same', return_sequences=False, kernel_initializer='he_normal'))(x)
    x = conv_block(x, num_filters)
    return x


# Hyperparameters
IMAGE_SIZE = 256
EPOCHS = 100
BATCH = 15
LR = 1e-4
model_path = "/home/kiran_shahi/dissertation/model/resnet_unet_single_frame.h5"
csv_path = "/home/kiran_shahi/dissertation/log/resnet_unet_single_frame.csv"


def resnet_unet():
    inputs = Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    encoder = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    for layer in encoder.layers:
        layer.trainable = False

    # encoder.summary()
    e1 = encoder.get_layer("input_1").output  ## (512 x 512 x 3)
    e2 = encoder.get_layer("conv1_relu").output  ## (256 x 256 x 64)
    e3 = encoder.get_layer("conv2_block3_out").output  ## (128 x 128 x 256)
    e4 = encoder.get_layer("conv3_block4_out").output  ## (64 x 64 x 512)

    # Bridge
    b1 = encoder.get_layer("conv4_block6_out").output  ## (32 x 32 x 1024)

    # Decoder
    d1 = decoder_block(b1, e4, 512)  ## (64 x 64)
    d2 = decoder_block(d1, e3, 256)  ## (128 x 128)
    d3 = decoder_block(d2, e2, 64)  ## (256 x 256)
    d4 = decoder_block(d3, e1, 3)  ## (512 x 512)

    outputs = Conv2D(1, (1, 1), 1, 'same', activation='sigmoid')(d4)
    return Model(inputs=inputs, outputs=outputs)



model = resnet_unet()
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(LR), metrics=[
    tf.keras.metrics.MeanIoU(num_classes=2),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.Precision()
])

callbacks = [
    ModelCheckpoint(model_path, monitor="val_loss", verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
    CSVLogger(csv_path),
    EarlyStopping(monitor='val_loss', patience=10)
]

ds = pd.read_csv("image_seg.csv")


def load_data(images, masks):
    train_x, test_x = train_test_split(images, test_size=0.2, random_state=42)
    train_y, test_y = train_test_split(masks, test_size=0.2, random_state=42)
    return (train_x, train_y), (test_x, test_y)


(train_x, train_y), (test_x, test_y) = load_data(ds['image'].tolist(), ds['mask'].tolist())

train_steps = len(train_x) // BATCH
if len(train_x) % BATCH != 0:
    train_steps += 1

valid_steps = len(test_x) // BATCH
if len(test_x) % BATCH != 0:
    valid_steps += 1


train_dataset = tf_dataset(train_x, train_y, batch=BATCH)
valid_dataset = tf_dataset(test_x, test_y, batch=BATCH)

model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    callbacks=callbacks
)
