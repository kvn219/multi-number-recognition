import tensorflow as tf
import os

tf.app.flags.DEFINE_string('name', 'demo', 'File pattern for task.')
tf.app.flags.DEFINE_string('mode', 'train', 'File pattern for type of task: train, test, or valid.')
tf.app.flags.DEFINE_string('log_name', 'run1', 'File pattern for logging.')
tf.app.flags.DEFINE_integer('num_epochs', 10, 'Num of epochs. Default is 10.')
tf.app.flags.DEFINE_integer('batch_size', 36, 'Batch size.')
tf.app.flags.DEFINE_integer('num_threads', 6, 'Num of threads used.  Default is 6.  Use at your own risk.')
tf.app.flags.DEFINE_bool('verbose', False, 'Print out model architecture')
tf.app.flags.DEFINE_float('lrate', 0.0001, 'Learning rate. Default value is 0.0001.')
tf.app.flags.DEFINE_bool('random_crop', False, 'Perform random crops')
tf.app.flags.DEFINE_bool('grayscale', False, 'Convert images into grayscale')

FLAGS = tf.app.flags.FLAGS


class SVHNParams():
    def __init__(self):
        self.name = FLAGS.name
        self.mode = FLAGS.mode
        self.log = FLAGS.log_name
        self.num_threads = FLAGS.num_threads

        # General parameters
        self.num_train = 223972
        self.num_test = 13068
        self.num_valid = 11782
        self.random_crop = FLAGS.random_crop
        self.grayscale = FLAGS.grayscale
        if self.grayscale:
            self.channel = 1
            self.pixels = 64 * 64 * self.channel
        else:
            self.channel = 3
            self.pixels = 64 * 64 * self.channel

        self.verbose = FLAGS.verbose
        self.lrate = FLAGS.lrate

        # Mode dependent parameters
        if self.mode == 'train':
            self.batch_size = 36
            self.is_training = True
            self.num_epochs = FLAGS.num_epochs

        elif self.mode == "test":
            self.batch_size = 13068
            self.is_training = False
            self.num_epochs = 1

        elif self.mode == "valid":
            self.batch_size = 11782
            self.is_training = False
            self.num_epochs = 1

        else:
            raise ValueError('Provide a mode: train, test, or valid')

        # Path info
        self.train_gz = "train.tar.gz"
        self.test_gz = "test.tar.gz"
        self.extra_gz = "extra.tar.gz"

        self.dir_path = os.path.abspath(".")
        self.data_dir = self.dir_path + "/data/"

        self.record_name = "/{}.tfrecords".format(self.mode)
        self.records_dir = self.dir_path + "/records/{}".format(self.mode)
        self.records_path = self.dir_path + self.records_dir + self.mode + self.record_name

        self.checkpoint_dir = self.dir_path + "/checkpoints/{}/".format(self.name)
        self.checkpoint_dir_path = self.dir_path + "/checkpoints/{}".format(self.name)

        self.results_dir = self.dir_path + "/results/{}/".format(self.name)
        self.results_path = self.results_dir + "{}-{}.csv".format(self.mode, self.name)

        self.data_url = "http://ufldl.stanford.edu/housenumbers/"

        # Ensure path for outputs
        if not os.path.exists(self.checkpoint_dir):
            tf.gfile.MakeDirs(self.checkpoint_dir)

        if not os.path.exists(self.results_dir):
            tf.gfile.MakeDirs(self.results_dir)
