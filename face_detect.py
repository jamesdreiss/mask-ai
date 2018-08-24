import os
import argparse
import numpy as np
import cv2
import dlib


path = os.path.join(os.path.dirname(os.path.realpath(__file__)))

face_detector = dlib.get_frontal_face_detector()

landmarks_model = os.path.join(path, 'models', 'shape_predictor_68_face_landmarks.dat')
landmarks_predictor = dlib.shape_predictor(landmarks_model)


def draw_polylines(face_img, polylines):
    """
    Draw polyline segments onto face bounding box image

    :param face_img: cropped bounding box of face
    :param polylines: polyline segments for
    :return: face_img_lined: bounding box of face with all polylines
    """

    face_img_lined = cv2.polylines(face_img, polylines, False, (0, 255, 255), 1, cv2.LINE_AA)

    return face_img_lined


def define_polylines(landmarks):
    """
    Define the polyline segments used to draw the line segments of the full face polyline

    :param landmarks: a full_object_detection object containing 68 face landmarks
    :return: polylines: list of line segments containing coordinates used to draw line segments of
    the full face polyline
    """

    # reference points for 68 point model: https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
    face_polyline_segments = [
        (27, 22, 21, 20, 19, 18, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22),
        (0, 36, 17, 37, 18, 38, 20), (38, 19),  # around eye
        (21, 38, 37, 36, 41, 40, 39, 38),  # eye
        (39, 27, 31, 40, 1, 31, 3, 48, 5, 58, 59, 48, 31, 39),  # cheek
        (21, 27, 28, 29, 30, 33, 32, 31, 51, 50, 49, 48), (31, 30, 35),  # keystone to upper lip
        (60, 61, 62, 63, 64, 65, 66, 67, 60),  # inner lip
        (8, 58, 57, 56, 8),  # chin
        (56, 55, 54, 53, 52, 51),  # lip
        (51, 33, 34, 35, 51), (35, 54),  # under nose
        (56, 11, 54, 13, 35, 15, 47, 35, 42, 27, 35),  # cheek
        (42, 47, 46, 45, 44, 43, 42),  # eye
        (16, 45, 26, 44, 25, 43, 24), (22, 43, 23)
    ]

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
    Expand boundaries percentage wise and make sure they fit within the source image pixel locations

    :param boundaries: tuple of face boundary box locations (left, top, right, bottom)
    :param img_shape: tuple of shape of source image (row_count, col_count, num_channels)
    :return: trimmed: a tuple of face boundary box locations (left, top, right, bottom) that fit within the
    boundaries of the source image
    """

    edited = list(boundaries)

    length = edited[3] - edited[1]
    width = edited[2] - edited[0]
    edited[0] = boundaries[0] - round(.50 * width)
    edited[1] = boundaries[1] - round(.50 * length)
    edited[2] = boundaries[2] + round(.50 * width)
    edited[3] = boundaries[3] + round(.50 * length)

    edited = [max(edited[0], 0), max(edited[1], 0), min(edited[2], img_shape[1]), min(edited[3], img_shape[0])]
    edited = tuple(edited)

    return edited


def rect_to_tuple(face):
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
    parser.add_argument('-d', '--dir', required=True)
    args = parser.parse_args()

    for filename in os.listdir(os.path.join(path, args.dir)):
        if filename.endswith('jpg'):
            img = load_img(os.path.join(path, args.dir, filename))
            faces_rect = faces_detect(img)
            if faces_rect:
                for idx, face_rect in enumerate(faces_rect):
                    landmarks = get_landmarks(img, face_rect)
                    polylines = define_polylines(landmarks)
                    img_lined = draw_polylines(img, polylines)
                    boundaries = rect_to_tuple(face_rect)
                    edited = edit_boundaries(boundaries, img.shape)
                    face_img = img_lined[edited[1]:edited[3], edited[0]:edited[2]]
                    cv2.imwrite(os.path.join(path, args.dir, 'faces', filename[:-4] + '_' + str(idx) + '.jpg'),
                                face_img)


if __name__ == '__main__':
    main()
