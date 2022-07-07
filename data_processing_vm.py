from distutils.dir_util import copy_tree

# root_path = '/home/kiran_shahi/dissertation/dataset'
# indoor_path = '/home/kiran_shahi/dissertation/human_centric/dataset/train/indoor'
# out_path = '/home/kiran_shahi/dissertation/human_centric/dataset/train/outdoor'

to_alpha = "/home/kiran_shahi/dissertation/dataset/alpha"
from_alpha = "/home/kiran_shahi/dissertation/VideoMatte240K_JPEG_SD/train/pha"

to_image = "/home/kiran_shahi/dissertation/dataset/image"
from_image = "/home/kiran_shahi/dissertation/VideoMatte240K_JPEG_SD/train/fgr"


copy_tree(from_alpha, to_alpha)
copy_tree(from_image, to_image)
