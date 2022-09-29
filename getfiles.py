import os

import pandas as pd


# img = '/home/kiran_shahi/dissertation/dataset/image'
# msk = '/home/kiran_shahi/dissertation/dataset/alpha'
# img = 'E:\\ds\\set1\\image'
# msk = 'E:\\ds\\set1\\alpha'


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


def save_csv(in_path, out_file):
    if not os.path.exists("/home/kiran_shahi/dissertation/csv_data"):
        os.makedirs("/home/kiran_shahi/dissertation/csv_data")

    image_list = get_files(os.path.join(in_path, 'image'))
    mask_list = get_files(os.path.join(in_path, 'alpha'))

    out_file = os.path.join("/home/kiran_shahi/dissertation/csv_data", out_file)
    df = pd.DataFrame({'image': image_list, 'mask': mask_list})
    df.to_csv(out_file, index=False)

    print("File saved to {}".format(out_file))


input_path = input("Enter dataset path: ")
output_file = input("Enter the csv file name to save your data: ")
batch_size = int(input("Enter a batch size: "))

save_csv(input_path, output_file)
