import os
from six.moves import urllib
import re
import numpy as np
import torch


class MovingMNIST(torch.utils.data.Dataset):
    """(Missing Moving) MNIST Dataset.

    Raw data are: http://www.cs.toronto.edu/~nitish/unsupervised_video/
    This dataset class is based on the tychovdo/MovingMNIST dataset:
    https://github.com/tychovdo/MovingMNIST

    This class inherits from torch.utils.data.Dataset, and can be used to
    construct a DataLoader object.

    Args:
        root (string): Root directory of dataset where ``processed/``
            and  ``raw`` directories will be created.
        train (bool, optional): If True, loads training data; otherwise test data.
        test_frac (float, optional): Fraction of examples to use in test set.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.


    Example:
        >>> from data import MovingMNIST
        >>> moving_mnist = MovingMNIST(train=True, download=True)
        >>> dt, x, y = moving_mnist[0]

        >>> missing_mnist =  MovingMNIST(train=True, download=True, missing=True)
        >>> dt, x, y = missing_mnist[0]
    """

    url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
    raw_folder = "raw"
    processed_folder = "processed"

    def __init__(
        self,
        root="data",
        train=True,
        test_frac=0.2,
        download=False,
        missing=False,
    ):
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.test_frac = test_frac
        self.missing = missing
        prefix = "missing_" if missing else ""
        self.train_file = f"{prefix}moving_mnist_train.pt"
        self.test_file = f"{prefix}moving_mnist_test.pt"
        self.path_prefix = os.path.join(self.root, self.processed_folder)
        self.train_path = os.path.join(self.path_prefix, self.train_file)
        self.test_path = os.path.join(self.path_prefix, self.test_file)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found."
                + " You can use download=True to download it"
            )

        if self.train:
            self.path = self.train_path
        else:
            self.path = self.test_path
        self.data = torch.load(self.path)

        self.dt_path = re.sub("\\.pt", "dt.pt", self.path)
        self.dt = torch.load(self.dt_path)

    def __getitem__(self, index):
        """
        Get time intervals, input image sequence, and output image to predict

        Args:
            index (int): Index

        Returns:
            tuple: (dt, seq, target) containing time intervals (dt), a sequence
                of inputs (seq), and an output (target)
        """
        dt = self.dt[index]
        seq, target = self.data[index, :10], self.data[index, -1]
        return dt, seq, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(self.train_path) and os.path.exists(
            self.test_path
        )

    def download(self):
        """Download the Moving MNIST data if it doesn't exist already."""
        if self._check_exists():
            return

        os.makedirs(os.path.join(self.root, self.raw_folder), exist_ok=True)
        os.makedirs(
            os.path.join(self.root, self.processed_folder), exist_ok=True
        )

        print("Downloading " + self.url)
        data = urllib.request.urlopen(self.url)
        filename = self.url.rpartition("/")[-1]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        with open(file_path, "wb") as f:
            f.write(data.read())

        # process and save as torch files
        print("Processing...")
        mnist_path = os.path.join(
            self.root, self.raw_folder, "mnist_test_seq.npy"
        )
        mnist_array = np.load(mnist_path).swapaxes(
            0, 1
        )  # reshape to (10000, 20, 64, 64)
        assert mnist_array.shape == (10000, 20, 64, 64)
        n_seq = mnist_array.shape[0]
        n_frames = mnist_array.shape[1]
        img_dim = mnist_array.shape[-1]
        desired_seq_len = 11
        assert 2 <= desired_seq_len <= n_frames
        split = int(self.test_frac * n_seq)

        def _get_idx():
            """ Generate a sequence of frame indices. """
            if self.missing:
                # Randomly excise frames
                ix = np.random.choice(n_frames, desired_seq_len, replace=False)
            else:
                # Choose consecutive frames with random offset
                max_offset = n_frames - desired_seq_len + 1
                ix = np.arange(desired_seq_len) + np.random.choice(max_offset)
            return np.sort(ix)

        idx = [_get_idx() for i in range(n_seq)]

        delta_t = [ix[1:] - ix[:-1] for ix in idx]
        delta_t = np.stack(delta_t)
        assert delta_t.shape == (n_seq, desired_seq_len - 1)

        mnist_list = [mnist_array[i, idx[i], :, :] for i in range(len(idx))]
        mnist_array = np.stack(mnist_list)
        assert mnist_array.shape == (n_seq, desired_seq_len, img_dim, img_dim)

        train_set = torch.from_numpy(mnist_array[:-split])
        train_delta_t = torch.from_numpy(delta_t[:-split])

        test_set = torch.from_numpy(mnist_array[-split:])
        test_delta_t = torch.from_numpy(delta_t[-split:])

        with open(self.train_path, "wb") as f:
            torch.save(train_set, f)

        with open(re.sub("\\.pt", "dt.pt", self.train_path), "wb") as f:
            torch.save(train_delta_t, f)

        with open(self.test_path, "wb") as f:
            torch.save(test_set, f)

        with open(re.sub("\\.pt", "dt.pt", self.test_path), "wb") as f:
            torch.save(test_delta_t, f)

        print("Done!")

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = "train" if self.train is True else "test"
        fmt_str += "    Train/test: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        return fmt_str
