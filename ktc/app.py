import numpy
import os
import argparse

import dsargparse

from ktc.cl import train
from ktc.cl import evaluate

def run(prog='python3 -m ktc'):
    parser = dsargparse.ArgumentParser(main=run, prog=prog)
    subparsers =  parser.add_subparsers(help='command')
    subparsers.add_parser(train.train)
    subparsers.add_parser(evaluate.evaluate)
    return parser.parse_and_run()


