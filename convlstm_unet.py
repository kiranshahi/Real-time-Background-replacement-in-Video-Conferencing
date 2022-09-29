import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, \
    Input, Conv3D, Dropout, TimeDistributed
from tensorflow.keras.models import Model

from data_frame import get_data as df_get_data
from data_image import get_data as di_get_data
from metrics import dice_loss, dice_coef, mad
import pandas as pd

lr = 1e-4
IMAGE_SIZE = 256
root_path = '/home/kiran_shahi/dissertation/'


def conv_block(inputs, num_filters):
    x = ConvLSTM2D(num_filters, 3, padding='same', return_sequences=True, dropout=0.8, recurrent_dropout=0.5)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = ConvLSTM2D(num_filters, 3, padding='same', return_sequences=True, dropout=0.8, recurrent_dropout=0.5)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = TimeDistributed(MaxPool2D((2, 2)))(x)
    return x, p


def decoder_block(inputs, skip_features, num_filters):
    x = TimeDistributed(Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same'))(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet():
    inputs = Input(shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")

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
    outputs = Conv3D(1, (3, 3, 3), padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs, name="U-Net")
    model.compile(loss=dice_loss,
                  optimizer=tf.keras.optimizers.Adam(lr),
                  metrics=[
                      dice_coef,
                      mad,
                      tf.keras.metrics.MeanSquaredError(),
                      tf.keras.metrics.MeanIoU(num_classes=2)
                  ])
    return model


def get_callback():
    csv_path = "/home/kiran_shahi/dissertation/log/convlstm_unet.csv"
    model_path = '/home/kiran_shahi/dissertation/model/convlstm_unet.h5'
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10),
        CSVLogger(csv_path),
        ModelCheckpoint(model_path, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
    ]
    return callbacks


def train_model(train_dataset, valid_dataset, checkpoint_path, batch_size, saved_model=None, epochs=100):
    if saved_model is not None:
        with CustomObjectScope({'mad': mad, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
            model = tf.keras.models.load_model(saved_model)
    else:
        model = build_unet()
    callbacks = get_callback()
    model.fit(train_dataset, validation_data=valid_dataset, epochs=epochs, callbacks=callbacks, batch_size=batch_size)


def call_train():
    # train_set = ['set1_train.csv', 'set2_train.csv']
    # valid_set = ['set1_valid.csv', 'set2_valid.csv']
    #
    # for count in range(3):
    #     if count == 0:
    #         saved_model = None
    #     else:
    #         saved_model = root_path + 'model/Set' + str(count) + '_model.h5'
    #     if count != 2:
    #         train_df = pd.read_csv(root_path + "csv_data/" + train_set[count])
    #         valid_df = pd.read_csv(root_path + "csv_data/" + valid_set[count])
    #         train_dataset, valid_dataset = df_get_data(train_df, valid_df, frame_size=15)
    #         train_model(train_dataset, valid_dataset, 'Set' + str(count + 1), batch_size=2, saved_model=saved_model,
    #                     epochs=100)
    #     else:
    #         (train_dataset, valid_dataset), (train_steps, valid_steps) = di_get_data(batch=8, sequence=True)
    #         train_model(train_dataset, valid_dataset, 'Set' + str(count + 1), batch_size=8, saved_model=saved_model,
    #                     epochs=10)

    # train_set = ['set1_train.csv', 'set2_train.csv']
    # valid_set = ['set1_valid.csv', 'set2_valid.csv']

    train_df = pd.read_csv(root_path + "csv_data/set1_train.csv")
    valid_df = pd.read_csv(root_path + "csv_data/set1_valid.csv")
    train_dataset, valid_dataset = df_get_data(train_df, valid_df, frame_size=15)
    train_model(train_dataset, valid_dataset, 'Set1', batch_size=1, saved_model=None, epochs=100)


call_train()
