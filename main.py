from model.bert.bert import Bert
from util import *
import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        gpu_number = 0  # Change this to the desired GPU number (e.g., 1, 2, etc.)
        tf.config.experimental.set_visible_devices(gpus[gpu_number], 'GPU')

        # Optional: Set memory growth to avoid memory allocation issues
        tf.config.experimental.set_memory_growth(gpus[gpu_number], True)

        print(f"Using GPU: {gpu_number}")
    except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)


def main():
    train_df, val_df = read_data()
    df = (train_df, val_df)
    model = Bert(df)
    # model.train()
    model.test()


if __name__ == '__main__':
    main()
