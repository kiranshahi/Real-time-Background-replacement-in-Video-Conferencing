import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import data_frame
import network as net

# Hyperparameters
EPOCHS = 100
BATCH = 2
LR = 1e-4
model_path = "/home/kiran_shahi/dissertation/model/resnet_unet_convlstm_aug_val.h5"
csv_path = "/home/kiran_shahi/dissertation/log/resnet_data_aug_convlstm_val.csv"

model = net.resnet_unet()
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

ds = pd.read_csv("image.csv")

train_dataset = data.tf_dataset(data.image_seq(ds['image'].tolist()), data.image_seq(ds['mask'].tolist()))

model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks)