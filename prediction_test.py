import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
from rembg import remove

image_path = (
    r"C:\Users\pc\Desktop\Sign language new\Test images\IMG_20251229_142412.jpg"
)

image = Image.open(image_path)
bgremoved_image = remove(image)
resized_image = bgremoved_image.resize((24, 24))
bw_image = resized_image.convert("L")
edged_image = bw_image.filter(ImageFilter.FIND_EDGES)

edged_image.show()

tf_model = tf.keras.models.load_model("second_model.keras")

img_array = np.array(edged_image).reshape(1, 24, 24, 1)

img_array = img_array / 255.0

prediction = tf_model.predict(img_array)

print(prediction)
