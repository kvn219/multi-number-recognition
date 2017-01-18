import os

class downloadParams():
    def __init__(self):
        self.dir = os.path.abspath(".")
        self.data_url = 'http://ufldl.stanford.edu/housenumbers/'
        self.train_filename = "train.tar.gz"
        self.test_filename = "test.tar.gz"
        self.extra_filename = "extra.tar.gz"