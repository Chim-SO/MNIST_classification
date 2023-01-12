from numpy.random import seed

seed(1)
from tensorflow import random, config
import tensorflow as tf

random.set_seed(1)
config.experimental.enable_op_determinism()
import random

random.seed(2)

from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

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

    # Preprocessing: scaling:
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Split dataset:
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

    # Display bars:
    fig, axs = plt.subplots(1, 2)
    unique, counts = np.unique(y_train, return_counts=True)
    axs[0].bar(unique, counts, width=0.4)
    axs[0].set_title('Train set')
    unique, counts = np.unique(y_val, return_counts=True)
    axs[1].bar(unique, counts, width=0.4)
    axs[1].set_title('Validation set')
    plt.show()

    # Create model:
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1],)))
    model.add(Dense(224, activation='sigmoid'))
    model.add(Dense(224, activation='sigmoid'))
    model.add(Dense(224, activation='sigmoid'))
    model.add(Dense(224, activation='sigmoid'))
    model.add(Dense(224, activation='sigmoid'))
    model.add(Dense(1, activation='relu'))
    print(model.summary())

    # Train:
    loss = 'mae'
    metric = 'mse'
    epochs = 200
    model.compile(loss=loss, optimizer='adam', metrics=[metric])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=1, validation_data=(x_val, y_val))

    # Display loss:
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig('output/single/loss.png', bbox_inches='tight')
    plt.show()
    # Display metric:
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'model {metric}')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig(f'output/single/{metric}.png', bbox_inches='tight')
    plt.show()

    # Evaluation:
    test_results = model.evaluate(x_test, y_test, verbose=1)
    print(f'Test set: - loss: {test_results[0]} - {metric}: {test_results[1]}')

    # Classification evaluation:
    pred_train = np.rint(np.clip(model.predict(x_train), 0, 9))
    pred_val = np.rint(np.clip(model.predict(x_val), 0, 9))
    pred_test = np.rint(np.clip(model.predict(x_test), 0, 9))
    print("Displaying other metrics:")
    print("\t\tAccuracy (%)\tPrecision (%)\tRecall (%)")
    print(
        f"Train:\t{round(accuracy_score(y_train, pred_train, normalize=True) * 100, 2)}\t\t\t"
        f"{round(precision_score(y_train, pred_train, average='macro') * 100, 2)}\t\t\t"
        f"{round(recall_score(y_train, pred_train, average='macro') * 100, 2)}")
    print(
        f"Val :\t{round(accuracy_score(y_val, pred_val, normalize=True) * 100, 2)}\t\t\t"
        f"{round(precision_score(y_val, pred_val, average='macro') * 100, 2)}\t\t\t"
        f"{round(recall_score(y_val, pred_val, average='macro') * 100, 2)}")
    print(
        f"Test:\t{round(accuracy_score(y_test, pred_test, normalize=True) * 100, 2)}\t\t\t"
        f"{round(precision_score(y_test, pred_test, average='macro') * 100, 2)}\t\t\t"
        f"{round(recall_score(y_test, pred_test, average='macro') * 100, 2)}")

    # Confusion matrix:
    ConfusionMatrixDisplay.from_predictions(y_val, pred_val, normalize='true')
    plt.savefig('output/single/confmat.png', bbox_inches='tight')
    plt.show()
