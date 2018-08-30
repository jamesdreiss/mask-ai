import os
import sys
import logging
import argparse
import cv2


path = os.path.join(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def process_video(vid):
    fps = 30
    vidsize = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('/Users/jamesdreiss/Downloads/FacialRecognition/test.mp4', fourcc, fps, vidsize)

    while vid.isOpened():
        ret, frame = vid.read()
        if ret is True:
            frame = cv2.flip(frame, 1)

            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vid', required=True)
    args = parser.parse_args()

    vid = cv2.VideoCapture(args.vid)
    process_video(vid)


if __name__ == '__main__':
    main()
