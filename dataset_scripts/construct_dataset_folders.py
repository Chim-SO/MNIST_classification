import idx2numpy
import os
import shutil
from PIL import Image


def create_folders(folders_names, path):
    # Create the path if it does not exist:
    if os.path.exists(path):
        # Delete directory:
        shutil.rmtree(path)
    else:
        # Create directory:
        os.makedirs(path)

    # Create folders if it exists recreate it for each class:
    for f_name in folders_names:
        os.makedirs(os.path.join(path, str(f_name)))


def to_class_folders(images, labels, path):
    # Create_folders:
    create_folders(set(labels), path)

    for id, image, label in zip(range(1, images.shape[0] + 1), images, labels):
        # save image to label directory:
        image = Image.fromarray(image)
        im_path = [path, str(label), str(id) + '.png']
        # print(os.path.join(*im_path))  # Uncomment it to print the file
        image.save(os.path.join(*im_path))


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

    to_class_folders(images_data, labels_data, os.path.join('../dataset_folders', dataset_name))
