import os
import sys
import logging
import argparse
import time
import cv2

import mask_ai


path = os.path.join(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def rotate_frame(frame, degrees):
    """
    Return rotated numpy array image

    :param frame: single frame of video as a numpy array
    :param degrees: degrees of rotation, 0 - 360
    :return: Rotated image as numpy array
    """

    rows, cols = frame.shape[:2]

    m = cv2.getRotationMatrix2D((cols / 2, rows / 2), degrees, 1)
    frame_rotated = cv2.warpAffine(frame, m, (cols, rows))

    return frame_rotated


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

            frame_mask = mask_ai.apply_mask(frame, polyline='full', colors=(255, 0, 255), line_thickness=3,
                                            detect_faces=False)
            if frame_mask is not None:
                vid_write.write(frame_mask)
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.imshow('frame', frame_mask)
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
