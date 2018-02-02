""" main module """
import argparse
import os
import skimage.io
import cv2
import valid
import align


def main():
    """ main function """
    parser = argparse.ArgumentParser(
        description='Classify image documents')
    parser.add_argument('input_path', type=str,
                        help='path to directory with documents images')
    args = parser.parse_args()
    input_path = args.input_path

    if not os.path.isdir(input_path):
        raise Exception('%s is not directory' % input_path)

    paths = os.listdir(input_path)

    if not paths:
        raise Exception('%s is empty directory' % input_path)

    paths = [os.path.join(input_path, f) for f in paths]

    are_files = [os.path.isfile(path) for path in paths]

    if not sum(are_files):
        raise Exception('no files in %s' % (input_path))

    from DocumentsClassifier import DocumentsClassificator

    classifier = DocumentsClassificator()

    for path in paths:
        if os.path.isfile(path):
            img = skimage.io.imread(path)
            cv_img = cv2.imread(path)
            class_id = classifier.document_class(img)
            print("%s is of class %d" % (os.path.basename(path), class_id))

            # TODO: compare [0] and [1] images
            _, rot_img, angle = align.main(cv_img)

            print('angle %f\n' % angle)
            # validate
            # print results


if __name__ == '__main__':
    main()
