import os

from PIL import Image, ImageFilter
from rembg import remove


def Basic_PreProcessing(dataset_path, output_path):
    folder_list = os.listdir(dataset_path)
    for i in range(len(folder_list)):
        train_valid = folder_list[i]
        class_list = os.listdir(dataset_path + "\\" + train_valid)
        for i2 in range(len(class_list)):
            letters = class_list[i2]
            image_list = os.listdir(dataset_path + "\\" + train_valid + "\\" + letters)
            for i3 in range(len(image_list)):
                image_path = (
                    dataset_path
                    + "\\"
                    + train_valid
                    + "\\"
                    + letters
                    + "\\"
                    + image_list[i3]
                )
                image = Image.open(image_path)
                bgremoved_image = remove(image)
                resized_image = bgremoved_image.resize((24, 24))
                bw_image = resized_image.convert("L")
                edged_image = bw_image.filter(ImageFilter.FIND_EDGES)
                output_path_train_valid = (
                    output_path
                    + "\\"
                    + train_valid
                    + "\\"
                    + letters
                    + "\\"
                    + "preprocessed_"
                    + image_list[i3]
                )
                edged_image.save(output_path_train_valid)


dataset_path = r"C:\Users\pc\Desktop\Sign language new\unprocessed_dataset"
output_path = r"C:\Users\pc\Desktop\Sign language new\preprocessed_dataset"

Basic_PreProcessing(dataset_path, output_path)
