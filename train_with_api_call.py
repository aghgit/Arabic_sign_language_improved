import json
import os

import couchdb
import tensorflow as tf
from tensorflow.keras import callbacks, layers, optimizers
from tensorflow.keras.utils import image_dataset_from_directory

from model import build_cnn

USER = "admin"
PASS = "admin"
DB_NAME = "model_configuration"
DOC_ID = "afadae3c50ca9931b350b76eb1000ff0"

URL = f"http://{USER}:{PASS}@127.0.0.1:5984/"

try:
    server = couchdb.Server(URL)
    db = server[DB_NAME]
    config = db[DOC_ID]
    print(f"Connection successful! Loaded config: {DOC_ID}")
except Exception as e:
    print(f"CouchDB Connection Error: {e}")
    raise

IMAGE_SIZE = config["image_size"]
BATCH_SIZE = config["batch_size"]
EPOCHS = config["epochs"]
LR = config["learning_rate"]
VAL_SPLIT = config["validation_split"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR_TRAIN = os.path.abspath(
    os.path.join(BASE_DIR, "..", config["dataset_dir_train"])
)
DATASET_DIR_VALID = os.path.abspath(
    os.path.join(BASE_DIR, "..", config["dataset_dir_valid"])
)
print("Dataset dir:", DATASET_DIR_TRAIN)
print("Exists:", os.path.exists(DATASET_DIR_TRAIN))

print("Dataset dir:", DATASET_DIR_VALID)
print("Exists:", os.path.exists(DATASET_DIR_VALID))

train_ds = image_dataset_from_directory(
    DATASET_DIR_TRAIN,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=42,
    color_mode="grayscale",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
)

val_ds = image_dataset_from_directory(
    DATASET_DIR_VALID,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=42,
    color_mode="grayscale",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
)

class_names = train_ds.class_names
num_classes = 31
print(f"{num_classes} Detected classes")

LABELS_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", config["labels_save_path"]))
os.makedirs(os.path.dirname(LABELS_PATH), exist_ok=True)

with open(LABELS_PATH, "w") as f:
    json.dump(class_names, f, indent=2)

normalization = layers.Rescaling(1.0 / 255)

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
        layers.RandomTranslation(0.05, 0.05),
    ]
)


def preprocess_train(x, y):
    x = normalization(x)
    x = data_augmentation(x)
    return x, y


def preprocess_val(x, y):
    return normalization(x), y


AUTOTUNE = tf.data.AUTOTUNE
train_ds = (
    train_ds.map(preprocess_train, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(1000)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds.map(preprocess_val, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
)

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)
model = build_cnn(input_shape, num_classes)

model.compile(
    optimizer=optimizers.Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

MODEL_SAVE_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", config["model_save_path"])
)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

cbs = [
    callbacks.EarlyStopping(
        monitor="val_loss", patience=config["patience"], restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1
    ),
    callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
    ),
]

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs)

print("Training done.")
