import gzip
import os
from six.moves import cPickle
import numpy as np
import theano


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        if "__file__" in globals():
            new_path = os.path.join(
                os.path.split(__file__)[0],
                "..",
                "data",
            )
            if os.path.isdir(new_path):
                new_path = os.path.join(new_path, dataset)
                if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
                    dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = cPickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # which rows correspond to an example. target is a
    # numpy.ndarray of 1 dimensions (vector) that have the same length as
    # the number of rows in the input. It should give the target
    # corresponding to the example with the same index in the input.

    def cast_dataset(data_xy):
        """Function that casts the target part of dataset to int32"""
        data_x, data_y = data_xy
        data_y = data_y.astype('int32')
        return data_x, data_y

    return [cast_dataset(data) for data in (train_set, valid_set, test_set)]
