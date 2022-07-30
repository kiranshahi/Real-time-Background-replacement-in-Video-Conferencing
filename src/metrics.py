from keras import backend


def mad(y_true, y_pred):
    return backend.mean(abs(y_pred - y_true))
