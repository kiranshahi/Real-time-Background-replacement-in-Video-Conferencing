import cv2
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 256
seq_size = 0


def read_image(images_path):
    images = []
    for path in images_path:
        path = path.decode()
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
        x = x / 255.0
        x = x.astype(np.float32)
        images.append(x)
    return np.array(images)


def read_mask(masks_path):
    masks = []
    for path in masks_path:
        path = path.decode()
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
        x = x / 255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=-1)
        masks.append(x)
    return np.array(masks)


def preprocess(image_path, mask_path):
    def f(img_path, msk_path):
        x = read_image(img_path)
        y = read_mask(msk_path)
        return x, y

    image, mask = tf.numpy_function(f, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([seq_size, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask.set_shape([seq_size, IMAGE_SIZE, IMAGE_SIZE, 1])
    return image, mask


def tf_dataset(images, masks, batch=4):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch, drop_remainder=True)
    dataset = dataset.prefetch(1)
    return dataset


def image_seq(images):
    img_list = []
    sub_list = []
    count = 0
    for image in images:
        count = count + 1
        if count != seq_size:
            sub_list.append(image)
        else:
            sub_list.append(image)
            img_list.append(sub_list)
            sub_list = []
            count = 0
    return img_list


def get_data(train_df, test_df, frame_size):
    seq_size=frame_size
    train_dataset = tf_dataset(image_seq(train_df['image'].tolist()), image_seq(train_df['mask'].tolist()))
    test_dataset = tf_dataset(image_seq(test_df['image'].tolist()), image_seq(test_df['mask'].tolist()))
    return train_dataset, test_dataset
