import os
import argparse
import numpy as np
import cv2
import dlib


path = os.path.join(os.path.dirname(os.path.realpath(__file__)))

face_detector = dlib.get_frontal_face_detector()

landmarks_model = os.path.join(path, 'models', 'shape_predictor_68_face_landmarks.dat')
landmarks_predictor = dlib.shape_predictor(landmarks_model)

# reference points for 68 point model: https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
# triangulation example: https://inst.eecs.berkeley.edu/~cs194-26/fa17/upload/files/proj4/cs194-26-abc/imgs/obama.png

# TODO: complete polyline segments
face_polyline_segments = [(27, 21, 19, 0, 1, 3, 5, 8, 11, 13, 15, 16, 26, 24, 22, 21, 38, 36, 0)]


def draw_polylines(face_img, polylines):
    """
    Draw polyline segments onto face bounding box image

    :param face_img: cropped bounding box of face
    :param polylines: polyline segments for
    :return: face_img_lined: bounding box of face with all polylines
    """

    face_img_lined = cv2.polylines(face_img, polylines, False, (0, 255, 255))

    return face_img_lined


def define_polylines(landmarks):
    """
    Define the polyline segments used to draw the line segments of the full face polyline

    :param landmarks: a full_object_detection object containing 68 face landmarks
    :return: polylines: list of line segments containing coordinates used to draw line segments of
    the full face polyline
    """

    polylines = []
    coors = [(p.x, p.y) for p in landmarks.parts()]
    for segment in face_polyline_segments:
        segment_coors = []
        for point in segment:
            segment_coors.append(coors[point])
        segment_coors = np.reshape(segment_coors, (-1, 2))
        polylines.append(segment_coors)

    return polylines


def get_landmarks(img, face_rect):
    """
    Get the coordinates for facial landmarks in the 68 point face model

    :param img: image array
    :param face_rect: dlib rectangle object with boundary boxe contaning a face
    :return: landmarks: dlib full_object_detection object containing the 68 face landmarks
    """

    landmarks = landmarks_predictor(img, face_rect)

    return landmarks


def edit_boundaries(boundaries, img_shape):
    """
    Edit face boundary box to fit within the source image pixel locations and expand top and bottom percentage wise

    :param boundaries: tuple of face boundary box locations (left, top, right, bottom)
    :param img_shape: tuple of shape of source image (row_count, col_count, num_channels)
    :return: trimmed: a tuple of face boundary box locations (left, top, right, bottom) that fit within the
    boundaries of the source image
    """

    trimmed = [max(boundaries[0], 0), max(boundaries[1], 0), min(boundaries[2], img_shape[1]),
               min(boundaries[3], img_shape[0])]

    length = boundaries[3] - boundaries[1]
    trimmed[1] = trimmed[1] - round(.25 * length)
    trimmed[3] = trimmed[3] + round(.10 * length)
    edited = tuple(trimmed)

    return edited


def rect_to_tuples(face):
    """
    Convert dlib rectangle object to a tuple of order (left, top, right, bottom)

    :param face: a dlib rectangle objet
    :return: boundaries: a tuple of face boundary box locations (left, top, right, bottom)
    """

    boundaries = (face.left(), face.top(), face.right(), face.bottom())

    return boundaries


def faces_detect(img):
    """
    Given a numpy array of an image, return a dlib rectangles object with any boundary boxes contaning faces

    :param img: source image converted to a numpy array
    :return: faces: dlib rectangles object with any boundary boxes contaning faces
    """

    faces = face_detector(img, 1)

    return faces


def load_img(img_file):
    """
    Given image location, load via Pillow's Image module

    :param img_file: location of image file
    :return: img: image as a multidimensional numpy array
    """

    img = cv2.imread(img_file, 1)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img', required=True)
    args = parser.parse_args()

    img = load_img(args.img)
    faces_rect = faces_detect(img)
    if faces_rect:
        for idx, face_rect in enumerate(faces_rect):
            landmarks = get_landmarks(img, face_rect)
            polylines = define_polylines(landmarks)
            img_lined = draw_polylines(img, polylines)
            boundaries = rect_to_tuples(face_rect)
            edited = edit_boundaries(boundaries, img.shape)
            face_img = img_lined[edited[1]:edited[3], edited[0]:edited[2]]
            cv2.imwrite(os.path.join(path, str(idx) + '.jpg'), face_img)


if __name__ == '__main__':
    main()
