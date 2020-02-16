"""Dataset reader and process"""

import os
import html
import string
import xml.etree.ElementTree as ET

import pre as pp
from functools import partial
from glob import glob
from multiprocessing import Pool

class Dataset():
    """Dataset class to read images and sentences from base (raw files)"""

    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.dataset = None
        self.partitions = ['train', 'valid', 'test']

    def read_partitions(self):
        """Read images and sentences from dataset"""

        self.dataset = getattr(self, f"_{self.name}")()

    def preprocess_partitions(self, input_size):
        """Preprocess images and sentences from partitions"""

        for y in self.partitions:
            arange = range(len(self.dataset[y]['gt']))

            for i in reversed(arange):
                text = pp.text_standardize(self.dataset[y]['gt'][i])

                if not self.check_text(text):
                    self.dataset[y]['gt'].pop(i)
                    self.dataset[y]['dt'].pop(i)
                    continue

                self.dataset[y]['gt'][i] = text.encode()

            pool = Pool()
            self.dataset[y]['dt'] = pool.map(partial(pp.preproc, input_size=input_size), self.dataset[y]['dt'])
            pool.close()
            pool.join()
