import os
import tkinter as tk
from tkinter import filedialog

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter, ImageTk
from rembg import remove

model = "second_model.keras"
tf_model = tf.keras.models.load_model(model)
class_names = os.listdir(
    r"C:\Users\pc\Desktop\Sign language new\preprocessed_dataset\train"
)


def Preprocessing_and_prediction(image_path):
    image = Image.open(image_path)
    bgremoved_image = remove(image)
    resized_image = bgremoved_image.resize((24, 24))
    bw_image = resized_image.convert("L")
    edged_image = bw_image.filter(ImageFilter.FIND_EDGES)
    img_array = (np.array(edged_image) / 255.0).astype("float32")
    prepared_input = np.expand_dims(img_array, axis=(0, -1))
    prediction = tf_model.predict(prepared_input)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_index]
    return edged_image, predicted_class_name


def open_result_window(image_path):
    processed_img, pred_index = Preprocessing_and_prediction(image_path)

    res_win = tk.Toplevel(root)
    res_win.title("Prediction Result")
    res_win.geometry("300x400")

    display_img = processed_img.resize((200, 200), Image.NEAREST)
    tk_img = ImageTk.PhotoImage(display_img)

    img_label = tk.Label(res_win, image=tk_img)
    img_label.image = tk_img
    img_label.pack(pady=20)

    result_label = tk.Label(
        res_win, text=f"Predicted Class: {pred_index}", font=("Arial", 14, "bold")
    )
    result_label.pack(pady=10)


def select_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    if file_path:
        open_result_window(file_path)


root = tk.Tk()
root.title("Sign Language Classifier")
root.geometry("400x200")

tk.Label(root, text="Sign Language Model Interface", font=("Arial", 12)).pack(pady=20)

select_btn = tk.Button(
    root,
    text="Choose Image",
    command=select_file,
    bg="#fab1b9",
    fg="white",
    font=("Arial", 10, "bold"),
    padx=20,
    pady=10,
)
select_btn.pack()

root.mainloop()
