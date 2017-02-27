#Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./straight_lines1.jpg "Original"
[image2]: ./straight_lines1_undistorted.jpg "Undistorted"
[image3]: ./binary_combo_example.jpg "Binary Example"
[image4]: ./straight_lines1_warped.jpg "Warp Example"
[image5]: ./color_fit_lines.jpg "Fit Visual"
[image6]: ./result.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## How to execute code

* Calibrate the camera `python calibrate.py`
* Process video `python process.py`

## Result video
[Youtube](https://www.youtube.com/watch?v=rz6koLB-t_Y)

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  


=)
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step has been isolated into the `calibrate.py` file. This python script has a single output, a `calibration.p` file, that contains the camera calibration matrix and distortion coefficients. Following python scripts will rely on the existence of this file to undistort camera images.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration matrix and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image2]


###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.

Once the camera has been calibrated and the camera matrix and distortion coefficients saved into `calibration.p`, those two pieces of information are used with the `cv2.undistort()` function to produce an image exempt of distortions.

```python
# process.py L.355
cv2.undistort(img, self.camera.mtx, self.camera.dist, None, self.camera.mtx)
```

| ![alt text][image1] Original | ![alt text][image2] Undistorted|
|-------|-----|

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (see `color_and_gradient_threshold` method in `process.py`).
For color threshold, I converted the image to HLS color space and applied a threshold on S component (see `color_threshold` method L.145 in `process.py`). As seen during the lectures the saturation component is good at picking up marked lines.
For gradient thresholds, I applied absolute sobel on x and y, and independently I applied magnitude and direction threshold (see `gradient_threshold` method L.134 in `process.py`). 

The binary output is the addition of those three thresholded results:

* S component (of HLS color space)
* absolute sobel x and y
* magnitude and direction

Here's an example of my output for this step. 

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `initialize()`, which appears in lines 80 in the file `process`.  The `initialize()` function takes as inputs an image (`img`), and will save once for all the matrix to apply the perspective transform and the inverted perspective transform.  I chose the hardcode the source and destination points in the following manner:

```
self.img_size = (img.shape[1], img.shape[0])
offset = [310, 10]
src = np.float32([
	[590, 450],
	[230, 670],
	[1070, 670],
	[692, 450]])
dst = np.float32([
	[offset[0], offset[1]],
	[offset[0], self.img_size[1] - offset[1]],
	[self.img_size[0] - offset[0], self.img_size[1] - offset[1]],
	[self.img_size[0] - offset[0], offset[1]]])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 590, 450      | 310, 10       | 
| 230, 670      | 310, 710      |
| 1127, 670     | 960, 710      |
| 692, 450      | 960, 10       |

I verified that my perspective transform was working by making sure that parallelism -of lanes lines- was kept from original to warped image.

| ![alt text][image1] Original | ![alt text][image4] Warped|
|-------|-----|

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used a histogram to verticaly sum up all the pixels of the lower half of the warped binary. I then took a maximum of the left and right of the histogram center. Those two points are used as starting point to discover lane lines. From bottom up, windows are used such as as many points are in the window and the window is centered. When moving to the next layer, next windows on y space, the new window can only move ever so slightly on the x space -to guarantee we detect lines-. See `find_windows()` L.161 in `process.py` file for implementation details.

![alt text][image5]

All point within those windows are then used used to fit the lane lines with a 2nd order polynomial (see L.311 in `process.py` file)

```python
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of curvature is calculated by the `calculate_curvature` method L.271 in `process.py`. The 2nd order polynomial processed during last step is in pixel space (birds view point). As radius is in real world meters, I had to re-calculate the polynomial (I approximated the conversion ratio by looking at U.S. regulations for lane width and eye balled the distance covered by my birds view point).

```
 # Define y-value where we want radius of curvature
 # I'll choose the maximum y-value, corresponding to the bottom of the image
 y_eval = np.max(self.right_line.besty) * self.ym_per_pix
 left_fit_cr = np.polyfit(self.left_line.besty * self.ym_per_pix, self.left_line.bestx * self.xm_per_pix, 2)
 # Calculate the new radii of curvature
 self.left_line.radius_of_curvature = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
```

To keep trace of the car position with respect to the lane center, I used the first lane lines pixels detected in y space (closer to the car) and assumed that the image was taken from the center of the car. The result is a simple difference between the center of the lanes and where the car stands whithin the lane (see L.227 in `process.py`).


```python
lane_center = (self.left_line.bestx[-1] + self.right_line.bestx[-1]) / 2
center_diff = (self.img_size[0] / 2 - lane_center) * self.xm_per_pix
```


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the `draw_lane` method L.251 in the `process.py` file. This is done by applying the inverted matrix and `cv2.warpPerspective` to go from birds view point to original view point.

Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Current implementation works well for the first video but it will very likely fail for challenge videos. First, the current color and gradient thresholds do a great job at picking up lane lines but unfortunately it's noisy and a few objects like car shadows or road patches are picked up. This noise can interfer the line detection algorithm. To mitigate this, we could add some consistency checks on the detected lines and or use it to drive the search, for instance we know the width of the lane, lines should be parallel, etc 

The video pipeline is simple. I start with a full search of lanes lines for the first frame of the video and then I assume only small changes from one frame to another will occur (in order to reduce the search space). This is a very simplistic approach, it works well on the first video but if the algorithm fails to keep track it will have a hard-time to recover. This could be mitigated by triggering a full search when no lanes lines have been detected over a few frames.