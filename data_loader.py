import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


class DataLoader:
    """Data Loader class. As a simple case, the model is tried on TinyImageNet. For larger datasets,
    you may need to adapt this class to use the Tensorflow Dataset API"""

    def __init__(self, batch_size, shuffle=False):
        self.X_train = None
        self.y_train = None
        self.img_mean = None
        self.train_data_len = 0

        self.X_val = None
        self.y_val = None
        self.val_data_len = 0

        self.X_test = None
        self.y_test = None
        self.test_data_len = 0

        self.shuffle = shuffle
        self.batch_size = batch_size

    def _unpickle(self, filename):
        filename = os.path.join("./dataset/cifar-10-batches-py", filename)
        with open(filename, 'rb') as file:
            data = pickle.load(file, encoding='bytes')
            return data

    def _load_data(self, filename):
        # Load the pickled data-file.
        data = self._unpickle(filename)

        # Get the raw images.
        images = data[b'data']

        # Get the class-numbers for each image. Convert to numpy-array.
        labels = np.array(data[b'labels'])

        return images, labels

    def load_data(self):
        # train
        files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
        data = [self._load_data(filename) for filename in files]

        images = [images for images, labels in data]
        images = np.concatenate(images, axis=0)

        images = images.reshape((images.shape[0], 3, 32, 32))
        images = images.transpose([0, 2, 3, 1])

        from scipy.misc import imresize
        images_reshape = np.zeros((images.shape[0], 224, 224, 3))
        for i, image in enumerate(images):
            images_reshape[i] = imresize(image, (224, 224))
        self.X_train = images_reshape

        labels = [labels for images, labels in data]
        self.y_train = np.concatenate(labels, axis=0)

        # test
        files = ["test_batch"]
        data = [self._load_data(filename) for filename in files]

        images = [images for images, labels in data]
        images = np.concatenate(images, axis=0)

        images = images.reshape((images.shape[0], 3, 32, 32))
        images = images.transpose([0, 2, 3, 1])

        from scipy.misc import imresize
        images_reshape = np.zeros((images.shape[0], 224, 224, 3))
        for i, image in enumerate(images):
            images_reshape[i] = imresize(image, (224, 224))
        self.X_val = images_reshape

        labels = [labels for images, labels in data]
        self.y_val = np.concatenate(labels, axis=0)

        # data statics
        self.train_data_len = len(self.X_train)
        self.val_data_len = len(self.X_val)
        img_height = 224
        img_width = 224
        num_channels = 3

        return img_height, img_width, num_channels, self.train_data_len, self.val_data_len

    def generate_batch(self, type='train'):
        """Generate batch from X_train/X_test and y_train/y_test using a python DataGenerator"""
        if type == 'train':
            # Training time!
            new_epoch = True
            start_idx = 0
            mask = None
            while True:
                if new_epoch:
                    start_idx = 0
                    if self.shuffle:
                        mask = np.random.choice(self.train_data_len, self.train_data_len, replace=False)
                    else:
                        mask = np.arange(self.train_data_len)
                    new_epoch = False

                # Batch mask selection
                X_batch = self.X_train[mask[start_idx:start_idx + self.batch_size]]
                y_batch = self.y_train[mask[start_idx:start_idx + self.batch_size]]
                start_idx += self.batch_size

                # Reset everything after the end of an epoch
                if start_idx >= self.train_data_len:
                    new_epoch = True
                    mask = None
                yield X_batch, y_batch
        elif type == 'test':
            # Testing time!
            start_idx = 0
            while True:
                # Batch mask selection
                X_batch = self.X_test[start_idx:start_idx + self.batch_size]
                y_batch = self.y_test[start_idx:start_idx + self.batch_size]
                start_idx += self.batch_size

                # Reset everything
                if start_idx >= self.test_data_len:
                    start_idx = 0
                yield X_batch, y_batch
        elif type == 'val':
            # Testing time!
            start_idx = 0
            while True:
                # Batch mask selection
                X_batch = self.X_val[start_idx:start_idx + self.batch_size]
                y_batch = self.y_val[start_idx:start_idx + self.batch_size]
                start_idx += self.batch_size

                # Reset everything
                if start_idx >= self.val_data_len:
                    start_idx = 0
                yield X_batch, y_batch
        else:
            raise ValueError("Please select a type from \'train\', \'val\', or \'test\'")
