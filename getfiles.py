import os
import glob
from pathlib import Path
import pandas as pd

batch_size = 15

img = '/home/kiran_shahi/dissertation/dataset/image'
msk = '/home/kiran_shahi/dissertation/dataset/alpha'


def get_files(file_dir):
    dirs = os.listdir(file_dir)
    images = []
    for dir in dirs:
        files = sorted(os.listdir(os.path.join(file_dir, dir)))
        file_count = 0
        print(dir)
        print(len(files))
        for file in files:
            images.append(os.path.join(file_dir, dir, file))
            file_count += 1
            if file_count == batch_size:
                break
    return images


images = get_files(img)
masks = get_files(msk)

df = pd.DataFrame({'image': images, 'mask': masks})
df.to_csv("image.csv", index=False)


