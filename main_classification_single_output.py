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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def split_dataset(dataset, train_frac=0.7):
    train = dataset.sample(frac=train_frac)
    val = dataset.drop(train.index)
    return train, val


if __name__ == '__main__':
    # read dataset:
    df = pd.read_csv('dataset_csv/train.csv')

    # split dataset to train validation:
    train, val = split_dataset(df)
    x_train, y_train = train.drop('label', axis=1) / 255., train['label']
    x_val, y_val = val.drop('label', axis=1) / 255., val['label']

    # Display bars:
    fig, axs = plt.subplots(1, 2)
    axs[0].bar([str(_) for _ in range(10)], train['label'].value_counts(), width=0.4)
    axs[0].set_title('Train set')
    axs[1].bar([str(_) for _ in range(10)], val['label'].value_counts(), width=0.4)
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
    test_df = pd.read_csv('dataset_csv/test.csv')
    x_test, y_test = test_df.drop('label', axis=1) / 255., test_df['label']
    test_results = model.evaluate(x_test, y_test, verbose=1)
    print(f'Test set: - loss: {test_results[0]} - {metric}: {test_results[1]}')

    # Classification evaluation:
    train_pred = np.rint(np.clip(model.predict(x_train), 0, 9))
    val_pred = np.rint(np.clip(model.predict(x_val), 0, 9))
    test_pred = np.rint(np.clip(model.predict(x_test), 0, 9))
    print("Displaying other metrics:")
    print("\t\tAccuracy (%)\tPrecision (%)\tRecall (%)")
    print(
        f"Train:\t{round(accuracy_score(y_train, train_pred, normalize=True) * 100, 2)}\t\t\t"
        f"{round(precision_score(y_train, train_pred, average='macro') * 100, 2)}\t\t\t"
        f"{round(recall_score(y_train, train_pred, average='macro') * 100, 2)}")
    print(
        f"Val :\t{round(accuracy_score(y_val, val_pred, normalize=True) * 100, 2)}\t\t\t"
        f"{round(precision_score(y_val, val_pred, average='macro') * 100, 2)}\t\t\t"
        f"{round(recall_score(y_val, val_pred, average='macro') * 100, 2)}")
    print(
        f"Test:\t{round(accuracy_score(y_test, test_pred, normalize=True) * 100, 2)}\t\t\t"
        f"{round(precision_score(y_test, test_pred, average='macro') * 100, 2)}\t\t\t"
        f"{round(recall_score(y_test, test_pred, average='macro') * 100, 2)}")

    # Confusion matrix:
    ConfusionMatrixDisplay.from_predictions(val['label'], val_pred, normalize='true')
    plt.savefig('output/single/confmat.png', bbox_inches='tight')
    plt.show()
