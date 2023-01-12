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
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def split_dataset(dataset, train_frac=0.7):
    train = dataset.sample(frac=train_frac)
    val = dataset.drop(train.index)
    return train, val


if __name__ == '__main__':
    # read dataset:
    df = pd.read_csv('dataset_csv/train.csv')
    n_labels = 10
    # split dataset to train validation:
    train, val = split_dataset(df)
    x_train, y_train = train.drop('label', axis=1) / 255., tf.keras.utils.to_categorical(train['label'], num_classes=n_labels)
    x_val, y_val = val.drop('label', axis=1) / 255., tf.keras.utils.to_categorical(val['label'], num_classes=n_labels)

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
    model.add(Dense(224, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(224, activation='relu'))
    model.add(Dropout(0.4))
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
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig('output/onehot/loss.png', bbox_inches='tight')
    plt.show()
    # Display metric:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history[f'val_accuracy'])
    plt.title(f'model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig(f'output/onehot/{metric}.png', bbox_inches='tight')
    plt.show()

    # Evaluation:
    test = pd.read_csv('dataset_csv/test.csv')
    x_test, y_test = test.drop('label', axis=1) / 255., tf.keras.utils.to_categorical(test['label'], num_classes=n_labels)
    test_results = model.evaluate(x_test, y_test, verbose=1)
    print(f'Test set: - loss: {test_results[0]} - {metric}: {test_results[1]}')

    # Classification evaluation:
    pred_train = np.argmax(model.predict(x_train), axis=1)
    pred_val = np.argmax(model.predict(x_val), axis=1)
    pred_test = np.argmax(model.predict(x_test), axis=1)
    print()
    print("Displaying other metrics:")
    print("\t\tAccuracy (%)\tPrecision (%)\tRecall (%)")
    print(
        f"Train:\t{round(accuracy_score(train['label'], pred_train, normalize=True) * 100, 2)}\t\t\t"
        f"{round(precision_score(train['label'], pred_train, average='macro') * 100, 2)}\t\t\t"
        f"{round(recall_score(train['label'], pred_train, average='macro') * 100, 2)}")
    print(
        f"Val :\t{round(accuracy_score(val['label'], pred_val, normalize=True) * 100, 2)}\t\t\t"
        f"{round(precision_score(val['label'], pred_val, average='macro') * 100, 2)}\t\t\t"
        f"{round(recall_score(val['label'], pred_val, average='macro') * 100, 2)}")
    print(
        f"Test:\t{round(accuracy_score(test['label'], pred_test, normalize=True) * 100, 2)}\t\t\t"
        f"{round(precision_score(test['label'], pred_test, average='macro') * 100, 2)}\t\t\t"
        f"{round(recall_score(test['label'], pred_test, average='macro') * 100, 2)}")

    # Confusion matrix:
    ConfusionMatrixDisplay.from_predictions(val['label'], pred_val, normalize='true')
    plt.savefig('output/onehot/confmat.png', bbox_inches='tight')
    plt.show()
