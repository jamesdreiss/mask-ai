import os
import sys
import logging
import argparse
import time
import numpy as np
import cv2

import mask_ai as ai


path = os.path.join(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def get_chromakey_background(img, colors_bgr=[64, 177, 0]):
    """
    Return chroma key background for image overlaying
    :param img: numpy array of image to create background from
    :param colors_bgr: color for background
    :return: background in specified chroma
    """
    color_array = np.array(colors_bgr, dtype=np.uint8)
    chromakey_bg = np.tile(color_array, img.shape[0] * img.shape[1]).reshape(img.shape[0], img.shape[1], 3)

    return chromakey_bg


def show_frame(frame):
    """
    Display a frame of a video
    :param frame: frame to show
    """
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    else:
        return True


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


def process_video(vid_name, vid_cap, vid_write, mask_type='full', colors_bgr=(0, 255, 255),
                  thickness=3, smooth_frames=15, recover=True, chromakey=False):
    """
    Process and save an OpenCV video capture object with an applied mask (one face per video)

    :param vid_name: name of input video
    :param vid_cap: VideoCapture object
    :param vid_write: VideoWriter object
    :param mask_type: 'full', 'partial', or 'dots'
    :param colors_bgr: list of blue, green, red color values
    :param thickness: thickness of polyline values or dots
    :param smooth_frames: number of previous frames of landmarks to averge with current landmarks
    :param recover: boolean to attempt to apply previous frame's masks to frames that error in face detection
    :param chromakey: boolean to apply mask to a single color background
    """

    start = time.time()
    landmarks_list = []
    counter = 0

    while vid_cap.isOpened():
        counter += 1
        ret, frame = vid_cap.read()
        if ret is True:
            logging.info('processing frame {} of {}'.format(counter, vid_name))

            frame_v = rotate_img(frame, 270)

            faces_rect = ai.faces_detect(frame_v)
            if faces_rect:
                landmarks = ai.get_landmarks(frame_v, faces_rect[0])
                landmark_coors = np.array([(p.x, p.y) for p in landmarks.parts()])
                if smooth_frames:
                    landmarks_list.append(landmark_coors)
                    if len(landmarks_list) > smooth_frames:
                        _ = landmarks_list.pop(0)
                    landmark_coors = ai.avg_landmarks(landmark_coors, landmarks_list)
                if chromakey is True:
                    frame_v = get_chromakey_background(frame_v)
                if mask_type == 'dots':
                    frame_mask = ai.draw_landmarks(frame_v, [tuple(c) for c in landmark_coors], colors_bgr, thickness)
                else:
                    polylines = ai.define_polylines(landmark_coors, ai.face_polyline_segments[mask_type])
                    frame_mask = ai.draw_polylines(frame_v, polylines, colors_bgr, thickness)
                frame_mask_h = rotate_img(frame_mask, 90)
                if frame_mask_h is not None:
                    vid_write.write(frame_mask_h)
                    show = show_frame(frame_mask_h)
                    if show is False:
                        break
                else:
                    logging.error('error processing frame {} of {}'.format(counter, vid_name))
                    break
            else:
                logging.error('could not detect face in frame {} of {}'.format(counter, vid_name))
                if recover:
                    logging.info('attempting to apply previous mask to current frame {} of {}'.format(counter, vid_name))
                    if chromakey is True:
                        frame_v = get_chromakey_background(frame_v)
                    if mask_type == 'dots':
                        frame_mask = ai.draw_landmarks(frame_v, [tuple(c) for c in landmark_coors], colors_bgr, thickness)
                    else:
                        frame_mask = ai.draw_polylines(frame_v, polylines, colors_bgr, thickness)
                    frame_mask_h = rotate_img(frame_mask, 90)
                    if frame_mask_h is not None:
                        vid_write.write(frame_mask_h)
                        show = show_frame(frame_mask_h)
                        if show is False:
                            break
                    else:
                        logging.error('error processing frame {} of {}'.format(counter, vid_name))
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
    parser.add_argument('-o', '--outfile', required=False)
    parser.add_argument('-fps', default=30, required=False)
    parser.add_argument('-m', '--mask_type', default='full', required=False,
                        choices=['full', 'partial', 'dots'],
                        help='type of mask to overlay on frames')
    parser.add_argument('-c', '--colors_bgr', default='0,255,255', required=False,
                        help='separate multiple colors with "-"')
    parser.add_argument('-th', '--thickness', default=3, required=False,
                        help='thickness of mask lines or dots')
    parser.add_argument('-sf', '--smooth_frames', default=15, required=False,
                        help='a smoothing technique: the number of previous frames to average with current frame')
    parser.add_argument('--chromakey', dest='chromakey', action='store_true')
    parser.add_argument('--no_chromakey', dest='chromakey', action='store_false')
    parser.set_defaults(chromakey=False)
    args = parser.parse_args()

    vid_cap = cv2.VideoCapture(args.infile)

    if args.outfile:
        outfile = args.outfile[:-4]
    else:
        outfile = args.infile[:-4]

    if args.chromakey:
        outfile += '_chromakey'

    outfile = '_'.join([outfile, 'm' + args.mask_type, 'c' + args.colors_bgr,
                        'th' + str(args.thickness), 'sf' + str(args.smooth_frames) + '.mp4'])

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vidsize = (int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vid_write = cv2.VideoWriter(outfile, fourcc, int(args.fps), vidsize)

    colors_bgr = [tuple(map(int, x.split(','))) for x in args.colors_bgr.split('-')]

    process_video(vid_name=args.infile, vid_cap=vid_cap, vid_write=vid_write, mask_type=args.mask_type,
                  colors_bgr=colors_bgr, thickness=int(args.thickness), smooth_frames=int(args.smooth_frames),
                  recover=True, chromakey=args.chromakey)


if __name__ == '__main__':
    main()
