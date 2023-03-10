import os

from numpy.random import seed

seed(1)
from tensorflow import random, config

random.set_seed(1)
config.experimental.enable_op_determinism()
import random

random.seed(2)

from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay, f1_score
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(dataset, train_frac=0.7):
    train = dataset.sample(frac=train_frac)
    val = dataset.drop(train.index)
    return train, val


os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == '__main__':
    # Read dataset:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(f"The training data shape: {x_train.shape}, its label shape: {y_train.shape}")
    print(f"The test data shape: {x_test.shape}, its label shape: {y_test.shape}")

    # Dimension transformation:
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    print(f"The training data shape becomes: {x_train.shape}, its label shape: {y_train.shape}")
    print(f"The test data shape becomes: {x_test.shape}, its label shape: {y_test.shape}")

    # One-hot encoding:
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    # Preprocessing: scaling:
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Split dataset:
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

    # Create model:
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1],)))
    model.add(Dense(224, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(224, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    print(model.summary())

    # Train:
    loss = 'categorical_crossentropy'
    metric = 'accuracy'
    epochs = 20
    model.compile(loss=loss, optimizer='adam', metrics=[metric])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=1, validation_data=(x_val, y_val))

    # Display loss:
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Onehot output model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()
    # Display metric:
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'Onehot output model {metric}')
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
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
    print("\t\tAccuracy (%)\tPrecision (%)\tRecall (%)\tF-measure (%)")
    print(
        f"Train:\t{round(accuracy_score(yy_train, pred_train, normalize=True) * 100, 2)}\t\t\t"
        f"{round(precision_score(yy_train, pred_train, average='macro') * 100, 2)}\t\t\t"
        f"{round(recall_score(yy_train, pred_train, average='macro') * 100, 2)}\t\t\t"
        f"{round(f1_score(yy_train, pred_train, average='macro') * 100, 2)}")
    print(
        f"Val :\t{round(accuracy_score(yy_val, pred_val, normalize=True) * 100, 2)}\t\t\t"
        f"{round(precision_score(yy_val, pred_val, average='macro') * 100, 2)}\t\t\t"
        f"{round(recall_score(yy_val, pred_val, average='macro') * 100, 2)}\t\t\t"
        f"{round(f1_score(yy_val, pred_val, average='macro') * 100, 2)}")
    print(
        f"Test:\t{round(accuracy_score(yy_test, pred_test, normalize=True) * 100, 2)}\t\t\t"
        f"{round(precision_score(yy_test, pred_test, average='macro') * 100, 2)}\t\t\t"
        f"{round(recall_score(yy_test, pred_test, average='macro') * 100, 2)}\t\t\t"
        f"{round(f1_score(yy_test, pred_test, average='macro') * 100, 2)}")

    # Confusion matrix:
    ConfusionMatrixDisplay.from_predictions(yy_val, pred_val, normalize='true')
    plt.show()
