import argparse
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from training.dataset import create_generators
from models.cnn_model import build_cnn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--img_size', type=int, default=128)
    p.add_argument('--save_dir', type=str, default='saved_model')
    return p.parse_args()


def plot_history(history, out_path):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title('Accuracy')

    plt.tight_layout()
    plt.savefig(out_path)


def main():
    args = parse_args()

    train_gen, val_gen = create_generators(
        args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )

    num_classes = len(train_gen.class_indices)
    model = build_cnn(input_shape=(args.img_size, args.img_size, 3), num_classes=num_classes)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(args.save_dir, 'model.h5'), save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss')
    ]

    os.makedirs(args.save_dir, exist_ok=True)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Save final SavedModel format
    model.save(os.path.join(args.save_dir, 'saved_model'))

    # Plot
    plot_history(history, os.path.join(args.save_dir, 'training_plot.png'))

    # Evaluate and print
    loss, acc = model.evaluate(val_gen)
    print(f'Validation loss: {loss:.4f}, acc: {acc:.4f}')


if __name__ == '__main__':
    main()
