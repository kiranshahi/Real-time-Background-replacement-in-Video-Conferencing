import os
import glob
from pathlib import Path


root_path = '/home/kiran_shahi/dissertation/dataset'
indoor_path = '/home/kiran_shahi/dissertation/human_centric/dataset/train/indoor'
out_path = '/home/kiran_shahi/dissertation/human_centric/dataset/train/outdoor'

def get_files(current_dir):
    dirs = os.listdir(os.path.join(current_dir))
    for dir in dirs:
        phas = glob.glob(os.path.join(current_dir, dir) + "/*.png")
        fgrs = glob.glob(os.path.join(current_dir, dir) + "/*.jpg")

        if len(phas) == len(fgrs) and len(phas) > 29:
            phas_path = os.path.join(root_path, "alpha", dir)
            if not os.path.exists(phas_path):
                os.makedirs(phas_path)

            for pha in phas:
                filename = os.path.basename(pha)
                if not os.path.exists(os.path.join(phas_path, filename)):
                    Path(pha).rename(os.path.join(phas_path, filename))

                fgr_path = os.path.join(root_path, "image", dir)
                if not os.path.exists(fgr_path):
                    os.makedirs(fgr_path)

                    for fgr in fgrs:
                        filename = os.path.basename(fgr)
                        if not os.path.exists(os.path.join(fgr_path, filename)):
                            Path(fgr).rename(os.path.join(fgr_path, filename))



get_files(indoor_path)
get_files(out_path)