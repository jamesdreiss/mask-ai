import os
import argparse
from PIL import Image
import numpy as np
import dlib


path = os.path.join(os.path.dirname(os.path.realpath(__file__)))

face_detector = dlib.get_frontal_face_detector()


def face_detect(img):
    faces = face_detector(img, 1)
    return faces


def load_img(img_file):
    img = Image.open(img_file)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img', required=True)
    args = parser.parse_args()

    img = load_img(args.img)
    img_array = np.asarray(img)
    faces = face_detect(img_array)
    for idx, f in enumerate(faces):
        region = img.crop((f.left(), f.top(), f.right(), f.bottom()))
        region.save(os.path.join(path, str(idx) + '.jpg'), 'JPEG')


if __name__ == '__main__':
    main()
