import idx2numpy
import pandas as pd
import numpy as np


def to_csv_one_output(images, labels):
    # Flatten images to arrays:
    n_images, im_size = images_data.shape[:2]
    images = images.flatten().reshape(n_images, im_size * im_size)

    # Create dataset: image as array , label
    dataset = []
    for image, label in zip(images, labels):
        image = image.tolist()
        image.append(label)  # append the label at the end.
        dataset.append(image)

    # Create header for the dataframe:
    header = [str(_ + 1) for _ in range(im_size * im_size)]
    header.append('label')

    return pd.DataFrame(dataset, columns=header)


def to_csv_one_hot(df):
    # Get one hot encoding of columns 'vehicleType'
    one_hot = pd.get_dummies(df['label'])
    # Drop column as it is now encoded
    df = df.drop('label', axis=1)
    # Join the encoded df
    return df.join(one_hot)


if __name__ == '__main__':
    # Files location:
    train_or_test = 't10k'  # train or t10k
    dataset_name = 'test'
    labels_file = f'dataset_idx/{train_or_test}-labels.idx1-ubyte'
    images_file = f'dataset_idx/{train_or_test}-images.idx3-ubyte'

    # Read data:
    labels_data = idx2numpy.convert_from_file(labels_file)
    images_data = idx2numpy.convert_from_file(images_file)

    # Display number of instances:
    n_images, im_size = images_data.shape[:2]
    n_labels = labels_data.shape[0]
    print(f"There is {n_images} images.")
    print(f"There is {n_labels} labels.")
    print(f"The images size is {im_size}.")

    # Create csv:
    df = to_csv_one_output(images_data, labels_data)
    df.to_csv(f'dataset_csv/{dataset_name}.csv', index=False)
