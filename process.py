import numpy as np
import cv2
import glob
import pickle
import os
from os.path import basename
from moviepy.editor import VideoFileClip


# %matplotlib qt

# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # Not detected count
        self.not_detected_count = 0
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        self.besty = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def update_detected(self, x, y, fit, fitx, fity, average_over=10):
        self.detected = True
        self.not_detected_count = 0
        self.allx = x
        self.ally = y
        self.diffs = self.current_fit - fit
        self.current_fit = fit
        self.besty = fity
        if self.bestx is None:
            self.bestx = fitx
            self.best_fit = fit
        else:
            self.bestx = ((self.bestx * average_over) + fitx) / (average_over + 1)
            self.best_fit = ((self.best_fit * average_over) + fit) / (average_over + 1)

    def update_not_detected(self):
        self.detected = False
        self.not_detected_count += 1


class Tracker:
    def __init__(self):
        # Calibration information
        self.camera = None
        # Left line
        self.left_line = Line()
        # Right line
        self.right_line = Line()
        # True if initialized
        self.initialized = False
        # Accepted image size
        self.img_size = None
        # Dashboard to Bird view matrix
        self.M = None
        # Bird view to dashboard matrix
        self.Minv = None
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 600  # meters per pixel in x dimension

    def reset_line_detection(self):
        """Reset line detection"""
        self.left_line = Line()
        self.right_line = Line()

    def initialize(self, img):
        if self.initialized:
            return

        self.img_size = (img.shape[1], img.shape[0])
        offset = [310, 10]
        src = np.float32([[590, 450], [230, 670], [1070, 670], [692, 450]])
        dst = np.float32(
            [[offset[0], offset[1]], [offset[0], self.img_size[1] - offset[1]],
             [self.img_size[0] - offset[0], self.img_size[1] - offset[1]], [self.img_size[0] - offset[0], offset[1]]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        self.initialized = True

    def abs_sobel_threshold(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel).astype(np.uint8)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output

    def magnitude_threshold(gray, sobel_kernel=3, thresh=(0, 255)):
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag).astype(np.uint8)
        binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
        return binary_output

    def direction_threshold(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)  # .astype(np.uint8)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        return binary_output

    def gradient_threshold(gray, ksize=3):
        # Apply each of the thresholding functions
        gradx_binary = Tracker.abs_sobel_threshold(gray, orient='x', sobel_kernel=ksize, thresh=(25, 255))
        grady_binary = Tracker.abs_sobel_threshold(gray, orient='y', sobel_kernel=ksize, thresh=(25, 255))
        mag_binary = Tracker.magnitude_threshold(gray, sobel_kernel=ksize, thresh=(50, 130))
        dir_binary = Tracker.direction_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))

        combined_binary = np.zeros_like(dir_binary)  # .astype(np.uint8)
        combined_binary[((gradx_binary == 1) & (grady_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return combined_binary

    def color_threshold(img, thresh=(170, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        s_binary = np.zeros_like(s_channel).astype(np.uint8)
        s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
        return s_binary

    def color_and_gradient_threshold(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gradient = Tracker.gradient_threshold(gray)
        color = Tracker.color_threshold(img)

        combined_binary = np.zeros_like(gradient)
        combined_binary[(gradient == 1) | (color == 1)] = 1
        return combined_binary

    def find_windows(binary_warped, filename, out_img=None):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            if out_img is not None:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def local_find_windows(binary_warped, left_fit, right_fit):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = (
            (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
                nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def draw_lane(self, binary_warped):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_line.bestx, self.left_line.besty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_line.bestx, self.right_line.besty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Draw lines
        if self.left_line.detected:
            color_warp[self.left_line.ally, self.left_line.allx] = [255, 0, 0]
        if self.right_line.detected:
            color_warp[self.right_line.ally, self.right_line.allx] = [0, 0, 255]
        return color_warp

    def calculate_curvature(self):
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.right_line.besty) * self.ym_per_pix
        left_fit_cr = np.polyfit(self.left_line.besty * self.ym_per_pix, self.left_line.bestx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.right_line.besty * self.ym_per_pix, self.right_line.bestx * self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        self.left_line.radius_of_curvature = ((1 + (
            2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        self.right_line.radius_of_curvature = ((1 + (
            2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    def process_image(self, img, filename=None):
        self.initialize(img)

        undistorted = self.undistort(img)
        if filename is not None:
            cv2.imwrite('output_images/%s_1_undistorted.jpg' % filename, undistorted)

        threshold_binary = Tracker.color_and_gradient_threshold(undistorted)
        if filename is not None:
            cv2.imwrite('output_images/%s_2_threshold.jpg' % filename, threshold_binary * 255)

        # Warp to birds view point
        binary_warped = cv2.warpPerspective(threshold_binary, self.M, self.img_size, flags=cv2.INTER_LINEAR)
        if filename is not None:
            cv2.imwrite('output_images/%s_3_warped.jpg' % filename, binary_warped * 255)

        # Create an output image to draw on and visualize processing on bird view
        detection_img = None
        if filename is not None:
            detection_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # Detect lines
        if self.left_line.detected and self.right_line.detected:
            leftx, lefty, rightx, righty = Tracker.local_find_windows(binary_warped, self.left_line.best_fit,
                                                                      self.right_line.best_fit)
        else:
            leftx, lefty, rightx, righty = Tracker.find_windows(binary_warped, filename, detection_img)

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # TODO Add some logic to detect if the data is coherent

        self.left_line.update_detected(leftx, lefty, left_fit, left_fitx, ploty)
        self.right_line.update_detected(rightx, righty, right_fit, right_fitx, ploty)

        # Draw lines
        if detection_img is not None:
            detection_img[lefty, leftx] = [0, 0, 255]
            detection_img[righty, rightx] = [255, 0, 0]

        # Save detection image
        if filename is not None:
            cv2.imwrite('output_images/%s_4_detection.jpg' % filename, detection_img * 255)

        self.calculate_curvature()
        cv2.putText(undistorted, 'Left curvature: %.2fm' % self.left_line.radius_of_curvature, (10, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(undistorted, 'Right curvature: %.2fm' % self.right_line.radius_of_curvature, (10, 75),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
        lane_center = (self.left_line.bestx[-1] + self.right_line.bestx[-1]) / 2
        center_diff = (self.img_size[0] / 2 - lane_center) * self.xm_per_pix
        side = 'right'
        if center_diff <= 0:
            side = 'left'
        cv2.putText(undistorted, '%.1fm %s of center' % (abs(center_diff), side), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

        color_lane_warped = self.draw_lane(binary_warped)
        color_lane = cv2.warpPerspective(color_lane_warped, self.Minv, self.img_size)

        # Combine the result with the original image
        result = cv2.addWeighted(undistorted, 1, color_lane, 0.3, 0)
        if filename is not None:
            cv2.imwrite('output_images/%s_5_result.jpg' % filename, result)
        return result

    def undistort(self, img):
        return cv2.undistort(img, self.camera.mtx, self.camera.dist, None, self.camera.mtx)


def process_test_images(path):
    global tracker
    # Get list of images
    images = glob.glob(path)
    for fname in images:
        img = cv2.imread(fname)
        filename = basename(fname)
        out_filename = os.path.splitext(filename)[0]
        # Start detection fresh
        tracker.reset_line_detection()
        tracker.process_image(img, out_filename)


def process_video(path):
    clip = VideoFileClip(path)
    output_clip = clip.fl_image(tracker.process_image)
    output_clip.write_videofile('out.mp4', audio=False)


tracker = Tracker()

if __name__ == "__main__":
    tracker.camera = pickle.load(open("calibration.p", "rb"))

    #process_video('project_video.mp4')
    process_test_images('test_images/*.jpg')

    cv2.destroyAllWindows()
