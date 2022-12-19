import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

if __name__ == '__main__':
    # Read df:
    df = pd.read_csv('../dataset_csv/train.csv')

    # Print some rows:
    print("Print some rows:")
    print(df.head())

    # Print the size of df:
    nrows, ncolumns = df.shape[:2]
    print(f"\nDataset size: {nrows} rows x {ncolumns} columns")
    print("Or we can print df.info():")
    print(df.info())

    # The number of instances of each class:
    print("The umber of instances of each class:")
    df_count = df['label'].value_counts()
    print(df_count)

    # Separate images from labels
    images = df.drop('label', axis=1)
    labels = df['label']

    # Display an image:
    element = 15
    im_shape = int(sqrt(images.shape[1]))
    plt.imshow(np.asarray(images.iloc[element]).reshape(im_shape, im_shape), cmap=plt.cm.binary)
    plt.title(labels.iloc[element])
    plt.show()

    # Plot the number of instances
    # plt.boxplot(labels)
    plt.bar([str(_) for _ in range(10)], df_count, width=0.4)
    plt.show()
