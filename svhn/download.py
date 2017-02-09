from __future__ import print_function
import os
import sys
import tarfile
import pickle

import numpy as np
from six.moves.urllib.request import urlretrieve
from time import sleep
import h5py

from data_config import downloadParams

np.random.seed(99)
last_percent_reported = None

params = downloadParams()


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download."""
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)
    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
        last_percent_reported = percent


def maybe_download(filename, force=False):
    """Download a file if not present."""
    # get root of filename by removing .tar.gz
    root = os.path.splitext(os.path.splitext(filename)[0])[0]
    # file directory
    file_dir = os.path.abspath(".")
    # complete file path
    path = "{}/{}".format(file_dir, filename)
    # check if path exists, if not download file
    if force or not os.path.exists(path):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(params.data_url + filename, filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    else:
        print("{} already present...skipping download".format(filename))
    statinfo = os.stat(file_dir)
    return filename


def maybe_extract(filename, force=False):
    # get root of filename by removing .tar.gz
    root = os.path.splitext(os.path.splitext(filename)[0])[0]
    # file directory
    file_dir = os.path.abspath(".")
    # complete file path
    path = "{}/{}".format(file_dir, filename)
    # check if path exists, if not download file
    if os.path.isdir(file_dir + "/" + root) and not force:
        print('{} folder already present...Skipping extraction of {}.'.format(root, filename))
    else:
        print('Extracting data for {}. This may take a while. Please wait.'.format(root))
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(".")
        tar.close()
    data_folders = root
    sleep(15)
    return data_folders


class DigitStructFile:
    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']

    def getName(self, n):
        """Returns the 'name' string for for the n(th) digitStruct."""
        return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value])

    def bboxHelper(self, attr):
        if (len(attr) > 1):
            attr = [self.inf[attr.value[j].item()].value[0][0]
                    for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr

    def getBbox(self, n):
        """Returns a dict of data for the n(th) bbox."""
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.bboxHelper(self.inf[bb]["height"])
        bbox['label'] = self.bboxHelper(self.inf[bb]["label"])
        bbox['left'] = self.bboxHelper(self.inf[bb]["left"])
        bbox['top'] = self.bboxHelper(self.inf[bb]["top"])
        bbox['width'] = self.bboxHelper(self.inf[bb]["width"])
        return bbox

    def getDigitStructure(self, n):
        s = self.getBbox(n)
        s['name'] = self.getName(n)
        return s

    # getAllDigitStructure returns all the digitStruct from the input file.
    def getAllDigitStructure(self):
        return [self.getDigitStructure(i) for i in range(len(self.digitStructName))]

    def getAllDigitStructure_ByDigit(self):
        pictDat = self.getAllDigitStructure()
        result = []
        structCnt = 1
        for i in range(len(pictDat)):
            item = {'filename': pictDat[i]["name"]}
            figures = []
            for j in range(len(pictDat[i]['height'])):
                figure = {}
                figure['height'] = pictDat[i]['height'][j]
                figure['label'] = pictDat[i]['label'][j]
                figure['left'] = pictDat[i]['left'][j]
                figure['top'] = pictDat[i]['top'][j]
                figure['width'] = pictDat[i]['width'][j]
                figures.append(figure)
            structCnt = structCnt + 1
            item['boxes'] = figures
            result.append(item)
        return result


def download_and_extract(file):
    filename = maybe_download(file)
    folder = maybe_extract(filename)
    return filename, folder


def get_digit_structure(folder):
    fin = os.path.join(folder, 'digitStruct.mat')
    dsf = DigitStructFile(fin)
    data = dsf.getAllDigitStructure_ByDigit()
    return data


def pickle_files(pickle_file, data):
    print("Pickling...{}".format(pickle_file))
    if not os.path.exists(pickle_file):
        try:
            f = open(pickle_file, 'wb')
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            f.close()
            print("Done pickling...{}".format(pickle_file))
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise
    else:
        print("{} already exists...skipping pickle".format(pickle_file))


def process(filename):
    name, folder = download_and_extract(filename)
    data = get_digit_structure(folder)
    root_name = os.path.splitext(os.path.splitext(filename)[0])[0]
    if not os.path.exists("./raw"):
        os.makedirs("./raw/")
    pickle_files("raw/" + root_name + ".pickle", data)


if __name__ == '__main__':
    process("train.tar.gz")
    process("test.tar.gz")
    process("extra.tar.gz")
