import os
import sys
import logging
import argparse
import time
import numpy as np
import cv2

import mask_ai


path = os.path.join(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def rotate_img(img, degrees):
    """
    Rotate an image without cropping it

    :param img: single frame of video as a numpy array
    :param degrees: degrees of desired counterclockwise rotation, 0 - 360
    :return: Rotated image as numpy array
    """

    (h, w) = img.shape[:2]
    (cx, cy) = (w // 2, h // 2)

    m = cv2.getRotationMatrix2D((cx, cy), degrees, 1.0)
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    m[0, 2] += (new_w / 2) - cx
    m[1, 2] += (new_h / 2) - cy

    img_rotated = cv2.warpAffine(img, m, (new_w, new_h))

    return img_rotated


def process_video(vid_name, vid_cap, vid_write):
    """
    Process and save an OpenCV video capture object

    :param vid_name: name of input video
    :param vid_cap: VideoCapture object
    :param vid_write: VideoWriter object
    """

    start = time.time()
    counter = 0

    while vid_cap.isOpened():
        counter += 1
        ret, frame = vid_cap.read()
        if ret is True:
            logging.info('processing frame {} of {}'.format(counter, vid_name))

            frame_v = rotate_img(frame, 270)
            frame_mask = mask_ai.apply_mask(frame_v, polyline='full', colors=(255, 0, 255), line_thickness=3,
                                            detect_faces=False)
            frame_mask_h = rotate_img(frame_mask, 90)

            if frame_mask_h is not None:
                vid_write.write(frame_mask_h)
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.imshow('frame', frame_mask_h)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                logging.error('could not detect face in frame {} of {}'.format(counter, vid_name))
                break
        else:
            break

    vid_cap.release()
    vid_write.release()
    cv2.destroyAllWindows()

    logging.info('processed {}, taking {} seconds'.format(vid_name, round(time.time() - start)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', required=True)
    parser.add_argument('-o', '--outfile', required=True)
    args = parser.parse_args()

    vid_cap = cv2.VideoCapture(args.infile)

    fps = 30
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vidsize = (int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vid_write = cv2.VideoWriter(args.outfile, fourcc, fps, vidsize)

    process_video(args.infile, vid_cap, vid_write)


if __name__ == '__main__':
    main()
