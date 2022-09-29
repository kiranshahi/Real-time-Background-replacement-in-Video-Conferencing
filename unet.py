import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, \
    Input
from tensorflow.keras.models import Model

from data_image import get_data
from metrics import dice_loss, dice_coef, mad

batch_size = 8
epochs = 100
lr = 1e-4
IMAGE_SIZE = 256


def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet():
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")

    """ Encoder """
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    b1 = conv_block(p4, 1024)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    """ Output """
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    return Model(inputs, outputs, name="U-Net")


def call_model():
    model = build_unet()
    csv_path = "/home/kiran_shahi/dissertation/log/unet.csv"
    model_path = '/home/kiran_shahi/dissertation/model/unet.h5'
    (train_dataset, valid_dataset), (train_steps, valid_steps) = get_data(batch_size, False)

    model.compile(
        loss=dice_loss,
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=[
            dice_coef,
            mad,
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.MeanIoU(num_classes=2)
        ]
    )

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=[
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
            EarlyStopping(monitor='val_loss', patience=10),
            CSVLogger(csv_path),
            ModelCheckpoint(model_path, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
        ])


call_model()
