import os

dataset_path = r"C:\Users\pc\Desktop\Sign language new\unprocessed_dataset\train"
output_path = r"C:\Users\pc\Desktop\Sign language new\preprocessed_dataset\valid"

subfolders = os.listdir(dataset_path)

for folder in subfolders:
    new_folder_path = os.path.join(output_path, folder)
    os.makedirs(new_folder_path)
