import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Activation, Conv2DTranspose, Input, Conv3D, \
    concatenate, TimeDistributed, Dropout

from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

from metrics import mad

LR = 1e-4


def conv_block(inputs, num_filters):
    x = ConvLSTM2D(filters=num_filters, kernel_size=(3, 3), padding='same', return_sequences=True,
                   kernel_initializer='he_normal', recurrent_dropout=0.3, dropout=0.3)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)
    return x


def decoder_block(inputs, skip_features, num_filters):
    x = TimeDistributed(Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same"))(inputs)
    x = concatenate([x, skip_features], axis=-1)
    x = conv_block(x, num_filters)
    return x


def resnet_unet():
    temporal_input = Input(shape=(None, 256, 256, 3), name="temporal_input")
    inputs = Input((256, 256, 3))
    encoder = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    for layer in encoder.layers:
        layer.trainable = False

    e2 = TimeDistributed(Model(encoder.input, encoder.get_layer("conv1_relu").output), name='time_distributed_model_1')(
        temporal_input)
    e3 = TimeDistributed(Model(encoder.input, encoder.get_layer("conv2_block3_out").output),
                         name='time_distributed_model_2')(temporal_input)
    e4 = TimeDistributed(Model(encoder.input, encoder.get_layer("conv3_block4_out").output),
                         name='time_distributed_model_3')(temporal_input)

    # Bridge
    b1 = TimeDistributed(Model(encoder.input, encoder.get_layer("conv4_block6_out").output),
                         name='time_distributed_model_4')(temporal_input)  ## (32 x 32 x 1024)

    # Decoder
    d1 = decoder_block(b1, e4, 512)  ## (64 x 64)
    d2 = decoder_block(d1, e3, 256)  ## (128 x 128)
    d3 = decoder_block(d2, e2, 64)  ## (256 x 256)

    outputs = TimeDistributed(Conv2DTranspose(15, (2, 2), strides=2, padding="same"))(d3)
    outputs = Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same")(outputs)

    model = Model(inputs=temporal_input, outputs=outputs)
    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(LR),
                  metrics=[
                      mad,
                      tf.keras.metrics.MeanSquaredError(),
                      tf.keras.metrics.MeanIoU(num_classes=2),
                      tf.keras.metrics.Recall(),
                      tf.keras.metrics.Precision()
                  ])
    return model
