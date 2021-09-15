import os
import pdb
import sys
import tensorflow as tf


def count(ds):
    size = 0
    for _ in ds: size += 1
    return size