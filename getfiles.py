import os

import pandas as pd

batch_size = 15

# img = '/home/kiran_shahi/dissertation/dataset/image'
# msk = '/home/kiran_shahi/dissertation/dataset/alpha'
img = 'E:\\ds\\set1\\image'
msk = 'E:\\ds\\set1\\alpha'


def get_files(file_dir):
    dirs = os.listdir(file_dir)
    images = []
    for current_dir in dirs:
        files = sorted(os.listdir(os.path.join(file_dir, current_dir)))
        file_count = 0
        for file in files:
            images.append(os.path.join(file_dir, current_dir, file))
            file_count += 1
            if file_count == batch_size:
                break
    return images


image_list = get_files(img)
mask_list = get_files(msk)

df = pd.DataFrame({'image': image_list, 'mask': mask_list})
df.to_csv("image_small_set1.csv", index=False)
