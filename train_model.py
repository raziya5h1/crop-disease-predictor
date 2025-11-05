# train_model.py
# Simple transfer-learning training script (example)
# Prepare dataset/ with train/ and val/ subfolders containing class folders.

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

DATA_DIR = "dataset"  # should contain train/ and val/ directories
BATCH = 16
IMG_SIZE = 224
EPOCHS = 6
MODEL_SAVE = "model/crop_disease_model.h5"

if not os.path.exists(DATA_DIR):
    print("Dataset directory not found. Please prepare dataset/train and dataset/val directories.")
    exit(1)

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1./255)

train_flow = train_gen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical'
)
val_flow = val_gen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical'
)

base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3),
                                         include_top=False, weights='imagenet')
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(train_flow.num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(train_flow, validation_data=val_flow, epochs=EPOCHS)
os.makedirs("model", exist_ok=True)
model.save(MODEL_SAVE)
print("Saved model to", MODEL_SAVE)
print("Class indices:", train_flow.class_indices)
