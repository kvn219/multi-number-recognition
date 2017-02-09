import pickle
import numpy as np
from PIL import Image
import os
import random
from tqdm import tqdm

SIZE_CROP = 64


def load_temp(dataset_name):
    filename = "raw/{}.pickle".format(dataset_name)
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        print('{} size:'.format(dataset_name), len(data))
    return data


def generate_dataset(data, folder):
    dataset = np.ndarray([len(data), SIZE_CROP, SIZE_CROP, 3], dtype=np.int8)
    labels = np.ones([len(data), 6], dtype=np.int8) * 10
    desc = "processing {} set".format(folder)

    for i in tqdm(np.arange(len(data)), desc=desc, leave=False):
        filename = data[i]['filename']
        fullname = os.path.join(folder, filename)
        try:
            im = Image.open(fullname)
        except FileNotFoundError:
            print("skipping {}".format(fullname))
            continue
        boxes = data[i]['boxes']
        num_digit = len(boxes)
        temp_label = [10] * 6
        temp_label[0] = num_digit

        # Get the individual bounding boxes
        top = np.ndarray([num_digit], dtype='float32')
        left = np.ndarray([num_digit], dtype='float32')
        height = np.ndarray([num_digit], dtype='float32')
        width = np.ndarray([num_digit], dtype='float32')
        for j in np.arange(num_digit):
            if j < 5:
                if boxes[j]['label'] == 10:
                    temp_label[j + 1] = 0
                else:
                    temp_label[j + 1] = boxes[j]['label']
            else:
                print('#', i, 'image has more than 5 digits.')
            top[j] = boxes[j]['top']
            left[j] = boxes[j]['left']
            height[j] = boxes[j]['height']
            width[j] = boxes[j]['width']

        # Get the bounding box surrounding all digits
        im_top = np.amin(top)
        im_left = np.amin(left)
        im_height = np.amax(top) + height[np.argmax(top)] - im_top
        im_width = np.amax(left) + width[np.argmax(left)] - im_left

        # Expanding by 30%
        im_bottom = np.amin([np.ceil(im_top + 1.3 * im_height), im.size[1]])
        im_right = np.amin([np.ceil(im_left + 1.3 * im_width), im.size[0]])
        im_top = np.amax([np.floor(im_top - 0.3 * im_height), 0])
        im_left = np.amax([np.floor(im_left - 0.3 * im_width), 0])

        # Cropping the expanded bounding box
        im = im.crop((int(im_left), int(im_top), int(im_right), int(im_bottom))).resize(
                [SIZE_CROP, SIZE_CROP],
                Image.ANTIALIAS)
        dataset[i, :, :, :] = im
        labels[i, :] = temp_label

    if folder == 'train':
        to_remove = 29929
        dataset = np.delete(dataset, to_remove, axis=0)
        labels = np.delete(labels, to_remove, axis=0)

    return dataset, labels


def get_dataset(folder):
    tmp_data = load_temp(folder)
    dataset, labels = generate_dataset(tmp_data, folder)
    del tmp_data
    return dataset, labels


def prep_train_val(train_dataset, train_labels, extra_dataset, extra_labels, validation_prop=0.05):
    random.seed(99)
    n_digits = 5
    valid_index = []
    valid_index2 = []
    train_index = []
    train_index2 = []

    for i in np.arange(n_digits):
        idx = np.where(train_labels[:, 0] == (i + 1))[0]
        numel_samples = len(idx)
        valid_index.extend(idx[:int(np.floor(numel_samples * validation_prop))].tolist())
        train_index.extend(idx[int(np.floor(numel_samples * validation_prop)):].tolist())

        idx = np.where(extra_labels[:, 0] == (i + 1))[0]
        numel_samples = len(idx)
        valid_index2.extend(idx[:int(np.floor(numel_samples * validation_prop))].tolist())
        train_index2.extend(idx[int(np.floor(numel_samples * validation_prop)):].tolist())

    # Shuffling the data
    random.shuffle(valid_index)
    random.shuffle(train_index)
    random.shuffle(valid_index2)
    random.shuffle(train_index2)

    valid_dataset = np.concatenate((extra_dataset[valid_index2, :, :, :], train_dataset[valid_index, :, :, :]),
                                   axis=0)
    valid_labels = np.concatenate((extra_labels[valid_index2, :], train_labels[valid_index, :]), axis=0)
    train_dataset_t = np.concatenate((extra_dataset[train_index2, :, :, :], train_dataset[train_index, :, :, :]),
                                     axis=0)
    train_labels_t = np.concatenate((extra_labels[train_index2, :], train_labels[train_index, :]), axis=0)
    del train_dataset, train_labels, valid_index, valid_index2, train_index, train_index2

    return train_dataset_t, train_labels_t, valid_dataset, valid_labels


if __name__ == '__main__':
    train_folders, test_folders, extra_folders = ('train', 'test', 'extra')
    train_dataset, train_labels = get_dataset(train_folders)
    test_dataset, test_labels = get_dataset(test_folders)
    extra_dataset, extra_labels = get_dataset(extra_folders)
    train_dataset_t, train_labels_t, valid_dataset, valid_labels = \
        prep_train_val(train_dataset, train_labels, extra_dataset, extra_labels)

    # make sure we have an "interim" directory
    if not os.path.exists("./interim"):
        os.makedirs("./interim/")

    # Save training set to .npz
    temp = np.reshape(train_dataset_t, (
        train_dataset_t.shape[0],
        train_dataset_t.shape[1] * train_dataset_t.shape[2] * train_dataset_t.shape[3]))
    temp = np.concatenate((train_labels_t, temp), axis=1)
    save_path = "./interim/train.npz"
    np.savez(save_path, train_dataset=train_dataset_t, train_labels=train_labels_t)
    print("Training set saved to {}".format(save_path))

    # Save valid/test set to .pickle
    save_path = './interim/test.npz'
    np.savez(save_path,
             test_dataset=test_dataset,
             test_labels=test_labels)
    print("Test set saved to {}".format(save_path))

    save_path = './interim/valid.npz'
    np.savez(
            save_path,
            valid_dataset=valid_dataset,
            valid_labels=valid_labels
    )
    print("Validation set saved to {}".format(save_path))
