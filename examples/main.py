import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# %matplotlib qt

# Choose a Sobel kernel size
ksize = 3  # Choose a larger odd number to smooth gradient measurements
gradx_threshold = (25, 80)
grady_threshold = (25, 80)
mag_threshold = (50, 130)
dir_threshold = (0.7, 1.3)  # threshold (0, np.pi/2)
s_threshold = (170, 255)


def calibrate(path):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(path)

    # Image size
    size = None

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if size == None:
            size = gray.shape[::-1]

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)

    return cv2.calibrateCamera(objpoints, imgpoints, size, None, None)


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return binary_output


def dir_thresh(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


def threshold(image):
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=gradx_threshold)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=grady_threshold)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, thresh=mag_threshold)
    dir_binary = dir_thresh(image, sobel_kernel=ksize, thresh=dir_threshold)

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


def color_and_gradient_threshold(img):
    thresholded = threshold(img)
    # cv2.imshow('img', thresholded)
    # cv2.waitKey(500)

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_threshold[0]) & (s_channel <= s_threshold[1])] = 1
    # cv2.imshow('img', s_binary * 255)
    # cv2.waitKey(500)

    combined_binary = np.zeros_like(thresholded)
    combined_binary[(s_binary == 1) | (thresholded == 1)] = 1
    return combined_binary


def process_image(img):
    img_size = (img.shape[1], img.shape[0])

    undistorded = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imshow('img', undistorded)
    cv2.waitKey(500)
    # from os.path import basename
    # cv2.imwrite('output_images/undistorded_%s.png' % basename(fname), undistorded)

    combined_binary = color_and_gradient_threshold(undistorded)
    cv2.imshow('img', combined_binary)
    cv2.waitKey(500)

    offset = [250, 10]
    src = np.float32([[605, 440], [200, 670], [1070, 670], [668, 440]])
    dst = np.float32(
        [[offset[0], offset[1]], [offset[0], img_size[1] - offset[1]],
         [img_size[0] - offset[0], img_size[1] - offset[1]], [img_size[0] - offset[0], offset[1]]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)
    cv2.imshow('img', warped)
    cv2.waitKey(2000)


if __name__ == "__main__":
    ret, mtx, dist, rvecs, tvecs = calibrate('camera_cal/calibration*.jpg')
    if not ret:
        print("Failed to calibrate camera")
        exit()
    # Get list of test images
    test_images = images = glob.glob('test_images/*.jpg')
    for fname in test_images:
        img = cv2.imread(fname)
        process_image(img)

    cv2.destroyAllWindows()
