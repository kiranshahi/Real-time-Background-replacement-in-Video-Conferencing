from glob import glob
import os
import pandas as pd

img = '/home/kiran_shahi/dissertation/old_dataset/new_data/train/image'
msk = '/home/kiran_shahi/dissertation/old_dataset/new_data/train/mask'


images = sorted(glob(os.path.join(img, "*png")))
masks = sorted(glob(os.path.join(msk, "*png")))

df = pd.DataFrame({'image': images, 'mask': masks})
df.to_csv("image_seg.csv", index=False)


