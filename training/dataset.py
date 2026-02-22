import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_generators(data_dir,
                      img_size=(128, 128),
                      batch_size=32,
                      validation_split=0.2,
                      seed=123):
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=validation_split
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='training',
        seed=seed
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',
        seed=seed
    )

    return train_gen, val_gen


def get_class_indices(data_dir):
    # expects each class as a subdirectory
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    return {c: i for i, c in enumerate(classes)}
