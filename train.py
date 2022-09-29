import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

import network as net
from data_frame import get_data as df_get_data
from data_image import get_data as di_get_data

from tensorflow.keras.utils import CustomObjectScope

from metrics import mad, iou, dice_coef, dice_loss

BATCH = 2
root_path = '/home/kiran_shahi/dissertation/'


def get_callback(checkpoint_path):
    csv_path = root_path + 'log/' + checkpoint_path + "_resnet_data_aug_val.csv"

    callbacks = [
        ModelCheckpoint(filepath=root_path + 'model/' + checkpoint_path + '_model.h5', monitor="val_loss",
                        verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    return callbacks


def train_model(train_dataset, valid_dataset, checkpoint_path, batch_size, saved_model=None, epochs=100):
    if saved_model is not None:
        with CustomObjectScope({'mad': mad, 'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
            model = tf.keras.models.load_model(saved_model)
    else:
        model = net.resnet_unet()
    callbacks = get_callback(checkpoint_path)
    model.fit(train_dataset, validation_data=valid_dataset, epochs=epochs, callbacks=callbacks, batch_size=batch_size)


def call_train():
    train_set = ['set1_train.csv', 'set2_train.csv']
    valid_set = ['set1_valid.csv', 'set2_valid.csv']

    for count in range(1, 3):
        if count == 0:
            saved_model = None
        else:
            saved_model = root_path + 'model/Set' + str(count) + '_model.h5'
        if count != 2:
            train_df = pd.read_csv(root_path + "csv_data/" + train_set[count])
            valid_df = pd.read_csv(root_path + "csv_data/" + valid_set[count])
            train_dataset, valid_dataset = df_get_data(train_df, valid_df, frame_size=15)
            train_model(train_dataset, valid_dataset, 'Set' + str(count + 1), batch_size=2, saved_model=saved_model,
                        epochs=100)
        else:
            (train_dataset, valid_dataset), (train_steps, valid_steps) = di_get_data(batch=8, sequence=True)
            train_model(train_dataset, valid_dataset, 'Set' + str(count + 1), batch_size=8, saved_model=saved_model,
                        epochs=100)


call_train()
