from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, UpSampling2D, Multiply, \
    Add


def get_lraspp(base_model, n_class=19):
    out_1_8 = base_model.get_layer("activation_1").output
    out_1_16 = base_model.get_layer("activation_13").output

    # branch1
    x1 = Conv2D(128, (1, 1))(out_1_16)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    # branch2
    # !!! important: the pool size and the strides of the next layer must be resized in case of input dimension
    #               equal to the one in the paper (224,224,3)

    x2 = AveragePooling2D(pool_size=(49, 49), strides=(16, 20))(out_1_16)
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
