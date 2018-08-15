import os
import argparse
from PIL import Image
import numpy as np
import dlib


path = os.path.join(os.path.dirname(os.path.realpath(__file__)))

face_detector = dlib.get_frontal_face_detector()


def get_boundaries(faces):
    boundaries = []
    for f in faces:
        boundaries.append((f.left(), f.top(), f.right(), f.bottom()))
    return boundaries


def trim_boundaries(boundaries, img_array_shape):
    trimmed = []
    for b in boundaries:
        trimmed.append((max(b[0], 0), max(b[1], 0), min(b[2], img_array_shape[1]), min(b[3], img_array_shape[0])))
    return trimmed


def faces_detect(img_array):
    faces = face_detector(img_array, 1)
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
    faces = faces_detect(img_array)
    boundaries = get_boundaries(faces)
    trimmed = trim_boundaries(boundaries, img_array.shape)
    for idx, t in enumerate(trimmed):
        region = img.crop(t)
        region.save(os.path.join(path, str(idx) + '.jpg'), 'JPEG')


if __name__ == '__main__':
    main()
