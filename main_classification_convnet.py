from numpy.random import seed

seed(1)
from tensorflow import random, config

random.set_seed(1)
config.experimental.enable_op_determinism()
import random

random.seed(2)

from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == '__main__':
    # read dataset:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(f"The training data shape: {x_train.shape}, its label shape: {y_train.shape}")
    print(f"The test data shape: {x_test.shape}, its label shape: {y_test.shape}")

    # Scale images to the [0, 1] range:
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Input dimension transformation:
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print(f"The training data shape: {x_train.shape}, its label shape: {y_train.shape}")
    print(f"The test data shape: {x_test.shape}, its label shape: {y_test.shape}")

    # Output dimension transformation:
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    # Preprocessing: scaling:
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Split dataset:
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

    # Create model:
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    print(model.summary())

    # Train:
    loss = 'categorical_crossentropy'
    metric = 'accuracy'
    epochs = 15
    model.compile(loss=loss, optimizer='adam', metrics=[metric])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=1, validation_data=(x_val, y_val))

    # Display loss:
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig('output/conv/conv_loss.png', bbox_inches='tight')
    plt.show()
    # Display metric:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history[f'val_accuracy'])
    plt.title(f'model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig(f'output/conv/conv_{metric}.png', bbox_inches='tight')
    plt.show()

    # Evaluation:
    test_results = model.evaluate(x_test, y_test, verbose=1)
    print(f'Test set: - loss: {test_results[0]} - {metric}: {test_results[1]}')

    # Classification evaluation:
    pred_train = np.argmax(model.predict(x_train), axis=1)
    pred_val = np.argmax(model.predict(x_val), axis=1)
    pred_test = np.argmax(model.predict(x_test), axis=1)
    yy_train = np.argmax(y_train, axis=1)
    yy_val = np.argmax(y_val, axis=1)
    yy_test = np.argmax(y_test, axis=1)
    print("Displaying other metrics:")
    print("\t\tAccuracy (%)\tPrecision (%)\tRecall (%)")
    print(
        f"Train:\t{round(accuracy_score(yy_train, pred_train, normalize=True) * 100, 2)}\t\t\t"
        f"{round(precision_score(yy_train, pred_train, average='macro') * 100, 2)}\t\t\t"
        f"{round(recall_score(yy_train, pred_train, average='macro') * 100, 2)}")
    print(
        f"Val :\t{round(accuracy_score(yy_val, pred_val, normalize=True) * 100, 2)}\t\t\t"
        f"{round(precision_score(yy_val, pred_val, average='macro') * 100, 2)}\t\t\t"
        f"{round(recall_score(yy_val, pred_val, average='macro') * 100, 2)}")
    print(
        f"Test:\t{round(accuracy_score(yy_test, pred_test, normalize=True) * 100, 2)}\t\t\t"
        f"{round(precision_score(yy_test, pred_test, average='macro') * 100, 2)}\t\t\t"
        f"{round(recall_score(yy_test, pred_test, average='macro') * 100, 2)}")

    # Confusion matrix:
    ConfusionMatrixDisplay.from_predictions(yy_val, pred_val, normalize='true')
    plt.savefig('output/conv/confmat.png', bbox_inches='tight')
    plt.show()
