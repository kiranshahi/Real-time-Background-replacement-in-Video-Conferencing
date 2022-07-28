import os
from glob import glob
import cv2
import numpy as np
import tensorflow as tf


def load_data():
    train_path = '/home/kiran_shahi/dissertation/old_dataset/new_data/train'
    test_path = '/home/kiran_shahi/dissertation/old_dataset/new_data/test'

    train_x = sorted(glob(os.path.join(train_path, "image/*")))
    train_y = sorted(glob(os.path.join(train_path, "mask/*")))

    test_x = sorted(glob(os.path.join(test_path, "image/*")))
    test_y = sorted(glob(os.path.join(test_path, "mask/*")))
    return (train_x, train_y), (test_x, test_y)


def read_image(images_path):
    x = cv2.imread(images_path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return x


def read_mask(masks_path):
    x = cv2.imread(masks_path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    x = np.expand_dims(x, axis=0)
    return x


def preprocess(image_path, mask_path):
    def f(img_path, msk_path):
        x = read_image(img_path.decode())
        y = read_mask(msk_path.decode())
        return x, y

    image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([1, 256, 256, 3])
    mask.set_shape([1, 256, 256, 1])
    return image, mask


def tf_dataset(images, masks, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset


def get_data(batch):
    (train_x, train_y), (test_x, test_y) = load_data()
    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(test_x, test_y, batch=batch)
    return train_dataset, valid_dataset
