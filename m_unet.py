from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, AveragePooling2D, UpSampling2D, \
    Multiply, Add, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import tensorflow as tf

from data_image import get_data
from metrics import dice_coef, dice_loss, mad

IMAGE_SIZE = 256
EPOCHS = 100
BATCH = 8
LR = 1e-4
smooth = 1e-15


def get_lraspp(base_model, n_class=19):
    out_1_8 = base_model.get_layer("re_lu_17").output
    out_1_16 = base_model.get_layer("re_lu_38").output

    # branch1
    x1 = Conv2D(128, (1, 1))(out_1_16)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    # branch2

    x2 = AveragePooling2D(pool_size=(4, 4), strides=(16, 20), data_format='channels_last')(out_1_16)
    x2 = Conv2D(128, (1, 1))(x2)
    x2 = Activation('sigmoid')(x2)
    x2 = UpSampling2D(size=(int(x1.shape[1]), int(x1.shape[2])), interpolation="bilinear")(x2)

    # branch3
    x3 = Conv2D(n_class, (1, 1))(out_1_8)

    # multiply
    m1 = Multiply()([x1, x2])
    m1 = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation="bilinear")(m1)
    m1 = Conv2D(n_class, (1, 1))(m1)

    # add
    m2 = Add()([m1, x3])
    return m2


def conv_block(inputs, num_filters):
    x = Conv2D(filters=num_filters, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def get_model():
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")
    encoder = MobileNetV3Large(input_tensor=inputs, weights="imagenet", include_top=False)

    s1 = encoder.get_layer('re_lu_1').output  # ([None, 128, 128, 16])
    s2 = encoder.get_layer('re_lu_6').output  # ([None, 64, 64, 72])
    s3 = encoder.get_layer('re_lu_12').output  # ([None, 32, 32, 120])

    # Backbone
    b = encoder.get_layer("re_lu_38").output  # ([None, 8, 8, 960])

    lrassp = get_lraspp(encoder, 120)

    d1 = decoder_block(b, lrassp, 120)
    d2 = decoder_block(d1, s3, 80)
    d3 = decoder_block(d2, s2, 40)
    d4 = decoder_block(d3, s1, 32)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    return Model(inputs, outputs)


def call_model():
    model = get_model()
    model.compile(
        loss=dice_loss,
        optimizer=tf.keras.optimizers.Adam(LR),
        metrics=[
            dice_coef,
            mad,
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.MeanIoU(num_classes=2)
        ]
    )

    csv_path = "/home/kiran_shahi/dissertation/log/m_unet.csv"
    model_path = '/home/kiran_shahi/dissertation/model/m_unet.h5'
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
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
