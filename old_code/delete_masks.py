import os
import shutil

def delete_masks(train_bin_path):
    if os.path.exists(train_bin_path):
        shutil.rmtree(train_bin_path)
        os.makedirs(train_bin_path, exist_ok=True)
        print("Deleted all files in train_bin_path and recreated the directory.")
    else:
        print("Directory does not exist.")