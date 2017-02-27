import argparse
import numpy as np
import cv2
import glob
import pickle
from common import *

chessboard_size = (9, 6)


def calibrate(path, visualize=False):
    """Calibrate camera"""
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    gray_size = None

    # Step through the list and search for chessboard corners
    for filename in glob.glob(path):
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_size = gray.shape[::-1]


        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

        # Draw and display the corners
        if visualize:
            image_corners = cv2.drawChessboardCorners(image, chessboard_size, corners, ret)
            cv2.imshow('image', image_corners)
            cv2.waitKey(500)

    return cv2.calibrateCamera(objpoints, imgpoints, gray_size, None, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calibrate camera')
    parser.add_argument('-o', '--output', default="calibration.p",
                        help='calibration output file')
    parser.add_argument('-d', '--display', action="store_true",
                        help='display detected rows and column on each image')
    args = parser.parse_args()

    ret, mtx, dist, _, _ = calibrate('camera_cal/calibration*.jpg', visualize=args.display)
    if not ret:
        print("Failed to calibrate camera")
        cv2.destroyAllWindows()
        exit()

    camera = Camera(mtx, dist)
    pickle.dump(camera, open(args.output, "wb"))
    print("Calibration saved to %s" % args.output)
    cv2.destroyAllWindows()
