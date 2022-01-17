import argparse
import os

import dsargparse


from . import trainvaltest
from . import crossval

def main(prog='python3 -m ktc.pre'):
    parser = dsargparse.ArgumentParser(main=main, prog=prog)
    subparsers = parser.add_subparsers(help='command')
    subparsers.add_parser(trainvaltest.split, add_arguments_auto=True)
    subparsers.add_parser(trainvaltest.remove_existing_folder, add_arguments_auto=True)
    subparsers.add_parser(crossval.crossvalDataGen, add_arguments_auto=True)
    return parser.parse_and_run()

if __name__ == '__main__':
    main()