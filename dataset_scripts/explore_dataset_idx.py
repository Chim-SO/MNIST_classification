import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Files location:
    train_or_test = 't10k'  # train or t10k
    labels_file = f'../dataset_idx/{train_or_test}-labels.idx1-ubyte'
    images_file = f'../dataset_idx/{train_or_test}-images.idx3-ubyte'

    # Read data:
    labels_data = idx2numpy.convert_from_file(labels_file)
    images_data = idx2numpy.convert_from_file(images_file)

    # Display number of instances:
    n_images, im_size = images_data.shape[:2]
    n_labels = labels_data.shape[0]
    print(f"There is {n_images} images.")
    print(f"There is {n_labels} labels.")
    print(f"The images size is {im_size}.")

    # Display an element:
    element = 10
    plt.imshow(images_data[element], cmap=plt.cm.binary)
    plt.title(labels_data[element])
    plt.savefig('explore.png', bbox_inches='tight')
    plt.show()
