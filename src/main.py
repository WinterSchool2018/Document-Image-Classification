""" main module """
import argparse
import os


def main():
    """ main function """
    parser = argparse.ArgumentParser(
        description='Check, verify filled documents')
    parser.add_argument('input_path', type=str, help='help')
    args = parser.parse_args()
    args = args.input_path

    for filename in os.listdir(input_path):
        # classify
        # align
        # validate
        # print results
        pass


if __name__ == '__main__':
    main()
