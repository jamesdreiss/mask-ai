import os
import argparse
import numpy as np
import cv2
import dlib
import logging
import random
import sys
import copy


path = os.path.join(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

face_detector = dlib.get_frontal_face_detector()

landmarks_model = os.path.join(path, 'models', 'shape_predictor_68_face_landmarks.dat')
landmarks_predictor = dlib.shape_predictor(landmarks_model)

# reference points for 68 point model: https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
face_polyline_segments = {
    'full':
        [
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
        ],
    'partial':
        [
            [x for x in range(0, 17)],
            (8, 48, 31, 30, 35, 54, 8),
            [x for x in range(60, 68)], (67, 60),
            [x for x in range(17, 27)], (0, 17), (16, 26),
            (31, 39, 21), (35, 42, 22)
        ]
}


def draw_landmarks(face_img, landmark_coors, colors=[(0, 255, 255)], thickness=1):
    """
    Draw landmarks points onto face bounding box image, with support for multi-colored dots

    :param face_img: cropped bounding box of face
    :param landmark_coors: list of tuples of (x, y) coordinates for the 68 face landmarks
    :param colors: list of blue, green, red color values
    :param thickness: line thickness
    :return: face_img_landmarked: bounding box of face with drawn landmark points
    """

    colors_rep = colors * (len(landmark_coors) // len(colors))
    colors_rep.extend(colors[:len(landmark_coors) % len(colors)])

    for idx, coor in enumerate(landmark_coors):
        cv2.circle(face_img, coor, 1, colors_rep[idx], thickness, cv2.LINE_AA)

    return face_img


def draw_polylines(face_img, polylines, colors=[(0, 255, 255)], thickness=1):
    """
    Draw polyline segments onto face bounding box image

    :param face_img: cropped bounding box of face
    :param polylines: coordinates of polyline segments
    :param colors: list blue, green, red color values, randomized if given > 1
    :param thickness: line thickness
    :return: face_img_lined: bounding box of face with all polylines
    """

    face_img_lined = cv2.polylines(face_img, polylines, False, random.choice(colors), thickness, cv2.LINE_AA)

    return face_img_lined


def define_polylines(landmark_coors, face_polyline_segment):
    """
    Define the polyline segments used to draw the line segments of the full face polyline

    :param landmark_coors: list of tuples of (x, y) coordinates for the 68 face landmarks
    :param face_polyline_segment: segment reference points for 68 face landmark model
    :return: polylines: list of line segments containing coordinates used to draw line segments of
    the full face polyline
    """

    polylines = []
    for segment in face_polyline_segment:
        segment_coors = []
        for point in segment:
            segment_coors.append(landmark_coors[point])
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
    edited[1] = boundaries[1] - round(.75 * length)
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


def avg_landmarks(landmark_coors, landmarks_list):
    """
    Average current landmark coordinates with a list of landmark coordinates

    :param landmark_coors: a numpy array of shape (68, 2) of average landmark points
    :param landmarks_list: list of numpy arrays (of shape (68, 2) of landmark points
    for averaging with current landmarks (a smoothing technique for video)
    :return: landmark_avg: numpy array of shape (68, 2) of average landmark points
    """

    landmark_avg = np.add(landmark_coors, sum(landmarks_list)) // (len(landmarks_list) + 1)

    return landmark_avg


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mask_type', default='full', required=False,
                        choices=['full', 'partial', 'dots'],
                        help='type of mask to overlay on frames')
    parser.add_argument('-c', '--colors_bgr', default='0,255,255', required=False,
                        help='separate multiple colors with "-"')
    parser.add_argument('-dth', '--dot_thickness', default=10, required=False)
    parser.add_argument('-lth', '--line_thickness', default=3, required=False)
    parser.add_argument('-d', '--dir', required=True)
    args = parser.parse_args()

    colors_bgr = [tuple(map(int, x.split(','))) for x in args.colors_bgr.split('-')]
    line_thickness = args.line_thickness
    dot_thickness = args.dot_thickness

    for filename in os.listdir(os.path.join(path, args.dir)):
        if filename.endswith(('jpg', 'JPG')):
            logging.info('processing: {}'.format(filename))
            img = load_img(os.path.join(path, args.dir, filename))
            faces_rect = faces_detect(img)
            if faces_rect:
                for idx, face_rect in enumerate(faces_rect):
                    logging.info('lines: {}, colors: {}'.format(filename, args.mask_type, args.colors_bgr))
                    boundaries = rect_to_tuple(face_rect)
                    bound_edit = edit_boundaries(boundaries, img.shape)
                    face_img = img[bound_edit[1]:bound_edit[3], bound_edit[0]:bound_edit[2]]

                    face_rect_edit = faces_detect(face_img)  # rerunning to obtain consistent polyline thickness
                    landmarks = get_landmarks(face_img, face_rect_edit[0])
                    landmark_coors = [(p.x, p.y) for p in landmarks.parts()]
                    if args.mask_type == 'dots':
                        img_dotted = draw_landmarks(copy.copy(face_img), landmark_coors, colors=colors_bgr,
                                                    thickness=dot_thickness)
                        cv2.imwrite(os.path.join(path, args.dir, 'faces',
                                                 '_'.join([filename[:-4], 'm' + args.mask_type, 'c' + args.colors_bgr,
                                                          'th' + str(args.line_thickness), str(idx)]) + '.jpg'), img_dotted)
                    else:
                        polylines = define_polylines(landmark_coors, face_polyline_segments[args.mask_type])

                        img_lined = draw_polylines(copy.copy(face_img), polylines, colors=colors_bgr,
                                                   thickness=line_thickness)
                        cv2.imwrite(os.path.join(path, args.dir, 'faces',
                                                 '_'.join([filename[:-4], 'm' + args.mask_type, 'c' + args.colors_bgr,
                                                          'th' + str(args.line_thickness), str(idx)]) + '.jpg'), img_lined)


if __name__ == '__main__':
    main()
