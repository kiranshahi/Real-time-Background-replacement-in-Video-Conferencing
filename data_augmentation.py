import os
from pathlib import Path
import cv2
from albumentations import HorizontalFlip, ChannelShuffle, CoarseDropout, CenterCrop, Rotate

batch_size = 30

# img = 'D:\\dissertation\\code\\background-replacement\\image'
# msk = 'D:\\dissertation\\code\\background-replacement\\alpha'

img = '/home/kiran_shahi/dissertation/dataset/image'
msk = '/home/kiran_shahi/dissertation/dataset/alpha'


def get_files(file_dir):
    dirs = os.listdir(file_dir)

    for current_dir in dirs:
        images = sorted(os.listdir(os.path.join(img, current_dir)))
        masks = sorted(os.listdir(os.path.join(msk, current_dir)))
        file_count = 0
        create_dir(current_dir)

        aug = Rotate(limit=45, p=1.0)

        for image, mask in zip(images, masks):
            img_path = os.path.join(img, current_dir, image)
            msk_path = os.path.join(msk, current_dir, mask)

            image_data = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mask_data = cv2.imread(msk_path, cv2.IMREAD_COLOR)

            # ###
            # Data Augmentation
            # ###

            aug_flip(image_data, mask_data, image, current_dir)
            aug_color(image_data, mask_data, image, current_dir)
            aug_channel(image_data, mask_data, image, current_dir)
            aug_coarse(image_data, mask_data, image, current_dir)

            aug_rotate(image_data, mask_data, image, current_dir, aug)
            aug_corp(image_data, mask_data, image, current_dir)

            file_count += 1
            if file_count == batch_size:
                break


def aug_flip(image, mask, file_name, current_dir):
    aug = HorizontalFlip(p=1.0)
    augmented = aug(image=image, mask=mask)
    image = augmented["image"]
    mask = augmented["mask"]
    save_image(current_dir + "_flip", file_name, image, mask)


def aug_color(image, mask, file_name, current_dir):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = mask
    save_image(current_dir + "_color", file_name, image, mask)


def aug_channel(image, mask, file_name, current_dir):
    aug = ChannelShuffle(p=1)
    augmented = aug(image=image, mask=mask)
    image = augmented['image']
    mask = augmented['mask']
    save_image(current_dir + "_channel", file_name, image, mask)


def aug_coarse(image, mask, file_name, current_dir):
    aug = CoarseDropout(p=1, min_holes=3, max_holes=10, max_height=32, max_width=32)
    augmented = aug(image=image, mask=mask)
    image = augmented['image']
    mask = augmented['mask']
    save_image(current_dir + "_coarse", file_name, image, mask)


def aug_rotate(image, mask, file_name, current_dir, aug):
    # aug = Rotate(limit=45, p=1.0)
    augmented = aug(image=image, mask=mask)
    image = augmented["image"]
    mask = augmented["mask"]
    save_image(current_dir + "_rotate", file_name, image, mask)


def aug_corp(image, mask, file_name, current_dir):
    H = 512
    W = 512
    try:
        """ Center Cropping """
        aug = CenterCrop(H, W, p=1.0)
        augmented = aug(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
    except Exception as e:
        image = cv2.resize(image, (W, H))
        mask = cv2.resize(mask, (W, H))
    save_image(current_dir + "_crop", file_name, image, mask)


def save_image(current_dir, name, image, mask):
    tmp_file_name = f"{Path(name).stem}.png"

    image_path = os.path.join(img, current_dir, tmp_file_name)
    mask_path = os.path.join(msk, current_dir, tmp_file_name)

    cv2.imwrite(image_path, image)
    cv2.imwrite(mask_path, mask)

    print(image_path)


def create_dir(dir_path):
    sub_paths = ['flip', 'color', 'channel', 'coarse', 'rotate', 'crop']

    for sub_path in sub_paths:
        img_sub_path = os.path.join(img, dir_path + "_" + sub_path)
        msk_sub_path = os.path.join(msk, dir_path + "_" + sub_path)

        if not os.path.exists(img_sub_path):
            os.makedirs(img_sub_path)

        if not os.path.exists(msk_sub_path):
            os.makedirs(msk_sub_path)


get_files(msk)
