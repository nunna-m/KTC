import numpy
import os
import argparse

import dsargparse

from ktc.cl import train
from ktc.cl import evaluate
from ktc.cl import split_train

def run(prog='python3 -m ktc'):
    parser = dsargparse.ArgumentParser(main=run, prog=prog)
    subparsers =  parser.add_subparsers(help='command')
    subparsers.add_parser(train.train, add_arguments_auto=True)
    subparsers.add_parser(split_train.split_train, add_arguments_auto=True)
    subparsers.add_parser(evaluate.evaluate, add_arguments_auto=True)
    return parser.parse_and_run()


