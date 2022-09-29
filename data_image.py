import os
from glob import glob
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

is_sequence = False


def load_data():
    train_path = '/home/kiran_shahi/dissertation/old_dataset/new_data/train'

    images = sorted(glob(os.path.join(train_path, "image/*")))
    masks = sorted(glob(os.path.join(train_path, "mask/*")))

    train_x, valid_x = train_test_split(images, test_size=0.2, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=0.2, random_state=42)
    return (train_x, train_y), (valid_x, valid_y)


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def read_image(images_path):
    x = cv2.imread(images_path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    x = x.astype(np.float32)
    if is_sequence:
        x = np.expand_dims(x, axis=0)
    return x


def read_mask(masks_path):
    x = cv2.imread(masks_path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    if is_sequence:
        x = np.expand_dims(x, axis=0)
    return x


def preprocess(image_path, mask_path):
    def f(img_path, msk_path):
        x = read_image(img_path.decode())
        y = read_mask(msk_path.decode())
        return x, y

    image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.float32])
    if is_sequence:
        image.set_shape([1, 256, 256, 3])
        mask.set_shape([1, 256, 256, 1])
    else:
        image.set_shape([256, 256, 3])
        mask.set_shape([256, 256, 1])
    return image, mask


def tf_dataset(images, masks, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset


def get_data(batch, sequence):
    global is_sequence
    is_sequence = sequence
    (train_x, train_y), (valid_x, valid_y) = load_data()

    train_x, train_y = shuffling(train_x, train_y)
    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    train_steps = len(train_x) // batch
    valid_steps = len(valid_x) // batch

    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1

    return (train_dataset, valid_dataset), (train_steps, valid_steps)
