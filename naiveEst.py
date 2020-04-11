from pathlib import Path

import numpy as np

INPUT_FILE = Path(__file__).absolute().parent / 'data.txt'
OUTPUT_FILE = Path(__file__).absolute().parent / 'output.txt'


def window(u: np.ndarray):
    """Window function.

    Args:
        u (np.ndarray): Input data.
    """
    if (np.abs(u) < 0.5).all():
        return 1
    else:
        return 0


class NaiveEst(object):
    def __init__(self, h=2):
        self.h = h
        self.n = 0
        self.m = 0
        self.instance_np = None

    def read_file(self, file_path: str):
        """Read file.
        Args:
            file_path (str): Path to the file to be loaded.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
            data = []

            # retrieve m and n
            meta_info = lines[0].split(',')
            self.n = int(meta_info[0])
            self.m = int(meta_info[1])

            # retrieve each line value
            for line in lines[1:]:
                row_data = list(map(float, line.split()))
                data.append(row_data)
            self.instance_np = np.array(data)

    def density(self, x: np.ndarray):
        """Calculate the density of the vector x and the data.

        Args:
            x (np.ndarray): Input vector. Should be a 1-d numpy array.

        Returns:
            float: Density probability.
        """
        assert x.ndim == 1, f'Dimension of the argument `x` expected to be 1, but got {x.ndim}'

        prob = 0
        volume = self.h ** self.m
        for x_i in self.instance_np:
            u = (x - x_i) / self.h
            prob += window(u)
        prob *= 1 / (self.n * volume)
        return prob

    def estimator(self):
        probs = []
        self.read_file(file_path=INPUT_FILE)
        for row in self.instance_np:
            probs.append(np.round(self.density(row), 2))
        self.write_file(probs, file_path=OUTPUT_FILE)

    @staticmethod
    def write_file(probs: list, file_path: str):
        with open(file_path, 'w+') as f:
            for prob in probs:
                f.write(f'{prob}\n')


if __name__ == '__main__':
    naive_est = NaiveEst(h=2)
    naive_est.read_file(INPUT_FILE)
    naive_est.estimator()
