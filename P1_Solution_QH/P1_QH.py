
# coding: utf-8

# # **Finding Lane Lines on the Road**
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below.
#
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
#
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
#
# **Note** If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".
#
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
#
# ---
#
# <figure>
#  <img src="line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p>
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p>
#  </figcaption>
# </figure>
#  <p></p>
# <figure>
#  <img src="laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p>
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p>
#  </figcaption>
# </figure>

# In[16]:

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().magic('matplotlib inline')



# In[17]:

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image


# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
#
# `cv2.inRange()` for color selection
# `cv2.fillPoly()` for regions selection
# `cv2.line()` to draw lines on an image given endpoints
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file
# `cv2.bitwise_and()` to apply a mask to an image
#
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[18]:

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[244, 175, 26], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    img_size_x = img.shape[1]
    img_size_y = img.shape[0]

    y_top = img_size_y*0.6
    # extend the line to the image bottom
    y_bottom = img_size_y

    # average for left and right lane mark
    #in case there is missing line detection
    line_avg_left = np.array([0,0,0,0])
    line_avg_right = np.array([0,0,0,0])
    #initialize lines_avg
    line_avg_left_initialized = 0
    line_avg_right_initialized = 0
    for line in lines:

        for x1,y1,x2,y2 in line:
            # calc slope. positive result means left lane mark
            if (y2-y1)/(x2 - x1) >0 :
                # [a + b for a, b in zip(list1, list2)]
                #TODO: Debug error from not any(line_avg_left)
                #TODO: is line an array instead of list?
                if not line_avg_left_initialized:
                    line_avg_left_initialized = 1
                    line_avg_left = np.array(line)
                else:
                    line_avg_left  = [ (new_value + avg_value)//2 for new_value, avg_value in zip(line, line_avg_left)]
            else:
                if not line_avg_right_initialized:
                    line_avg_right_initialized = 1
                    line_avg_right = np.array(line)
                else:
                    line_avg_right  = [ (new_value + avg_value)//2 for new_value, avg_value in zip(line, line_avg_right)]


    #ignore undetected side lines
    lines_averged = [line_avg_left, line_avg_right]
    if not line_avg_left_initialized:
        lines_averged = [line_avg_right]
    if not line_avg_right_initialized:
        lines_averged = [line_avg_left]


    # extrapolate for bottom x
    # xx = ((Xh - Xl)/(Yh-Yl) )*(yy - Yh) +  Xh
    for avg_line in lines_averged:
        for x1,y1,x2,y2 in avg_line:

            if y2 < y1:
                # (x2, y2) is a higher point
                Yh = y2
                Xh = x2
                Yl = y1
                Xl = x1
            else:
                Yh = y1
                Xh = x1
                Yl = y2
                Xl = x2

            # divide by zero condition
            if Yh == Yl:
                # TODO: horizontal lines causes error in extra.mp4 at 45th of 251 frame
                # print('error', 'y1= ',y2,' y2 =',y1)
                x_top = x1
                x_bottom = x2
                y_top = y1
                y_bottom = y2
            else:
                x_top = ((Xh - Xl)/(Yh-Yl) )*(y_top - Yh) +  Xh
                x_bottom = ((Xh - Xl)/(Yh-Yl) )*(y_bottom - Yh) +  Xh

            # ignore curve  condition
            if (x_top > 0) and (x_top < img_size_x) and (x_bottom > 0) and (x_bottom<img_size_x):
                cv2.line(img, (int(x_top), int(y_top)), (int(x_bottom), int(y_bottom)), color, thickness)



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """

    # lines – Output vector of lines. Each line is represented by a 4-element vector (x_1, y_1, x_2, y_2) ,
    #         where  (x_1,y_1) and  (x_2, y_2) are the ending points of each detected line segment.
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# ## Test on Images ==> QH: Save to "result_test_images\"
#
# Now you should build your pipeline to work on the images in the directory "test_images"
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[19]:
DEBUG_SWITCH = 1

# @param apex_portion range : 0~1.0 the percentage of relative position of apex, e.g., 0.2 is at 0.2*x
def QH_Region_GenTriangleVertices(image, apex_x_portion , apex_y_portion):

    bottom_portion = 0.1

    img_num_of_row = image.shape[0]
    img_num_of_col = image.shape[1]
    img_size_x = img_num_of_col
    img_size_y = img_num_of_row

    tup_botm_right = (img_size_x*(1-bottom_portion), img_size_y)
    tup_botm_left = (img_size_x*bottom_portion,img_size_y)
    tup_apex = (img_size_x*apex_x_portion, img_size_y*apex_y_portion)

    vertices = np.array([[tup_botm_left,tup_apex, tup_botm_right]], dtype=np.int32)

    return vertices

# @return bool index for numpy array
def QH_ImageFilter_Color_FindPixelIdx(img, thres_red,thres_green,thres_blue):

    rgb_threshold = [thres_red, thres_green, thres_blue]

    img_table_pixel_disable =   (img[:,:,0] < rgb_threshold[0] ) | \
                              (img[:,:,1] < rgb_threshold[1] ) | \
                              (img[:,:,2] < rgb_threshold[2] )

    return (~img_table_pixel_disable)

# @return image in numpy array of cropped region
def QH_ImageFilter_RegionCrop(img):
    vertices = QH_Region_GenTriangleVertices(img, 0.5, 0.5)
    img_cropped_region = region_of_interest(img, vertices)
    return img_cropped_region


def QH_process_image_RegionColorFilter(image):


    img_cropped_region = QH_ImageFilter_RegionCrop(image)


    color_threshold_red = 160
    color_threshold_green = 160
    color_threshold_blue = 0

    img_target_pixel_index = QH_ImageFilter_Color_FindPixelIdx(img_cropped_region, color_threshold_red, color_threshold_green, color_threshold_blue)


    img_cropped_region_color= np.copy(img_cropped_region)
    img_cropped_region_color[~img_target_pixel_index] = [0,0,0] #Boolean or “mask” index arrays

    # mark the targets
    img_marked_region_color= np.copy(img_cropped_region)
    img_marked_region_color[img_target_pixel_index] = [255,0,0] #Boolean or “mask” index arrays

    # mark the targets on the full image
    img_marked_color= np.copy(image)
    img_marked_color[img_target_pixel_index] = [255,0,0] #Boolean or “mask” index arrays

    # mark the targets on the full image
    img_colorFiltered= np.copy(image)
    img_colorFiltered[~img_target_pixel_index] = [0,0,0] # only show the filtered color

    result = img_marked_color

    # result = img_colorFiltered
    return result


def QH_process_image_HoughFilter(img):


    img_gray = grayscale(img)

    # higher size removes the false line from the car
    kernal_size = 7
    img_gray_blur = gaussian_blur(img_gray, kernal_size)

    # higher threshold removes lines between road and dirt ground
    img_gray_blur_canny = canny(img_gray_blur,130,200)

    img_gray_blur_canny_crop = QH_ImageFilter_RegionCrop(img_gray_blur_canny)


    if DEBUG_SWITCH:
        plt.imshow(img_gray_blur_canny_crop, cmap='gray')
        plt.show()

    rho = 2
    theta = np.pi/180
    vote_threshold = 20
    min_line_len = 40
    max_line_gap = 5
    img_gray_blur_canny_crop_hough = hough_lines(img_gray_blur_canny_crop, rho, theta, vote_threshold, min_line_len, max_line_gap)

    img_3chnn_hough_red = np.dstack((img_gray_blur_canny_crop_hough, np.zeros_like(img_gray_blur_canny_crop_hough), np.zeros_like(img_gray_blur_canny_crop_hough)))

    img_3chnn_canny= np.dstack((img_gray_blur_canny, img_gray_blur_canny, img_gray_blur_canny))

    if DEBUG_SWITCH:
        plt.imshow(weighted_img(img_3chnn_hough_red, img_3chnn_canny))
        plt.show()

    result = weighted_img(img_3chnn_hough_red,img)

    # plt.imshow(result)

    return result

def QH_process_image(image):
  # return QH_process_image_RegionColorFilter(image)
  return QH_process_image_HoughFilter(image)

import os
test_dir = "test_images/"
file_list = os.listdir(test_dir)

if not os.path.exists('result_'+test_dir):
    os.makedirs('result_'+test_dir)

for file in file_list:
    image = mpimg.imread( test_dir+file)
    img_processed = QH_process_image(image)
    mpimg.imsave('result_'+test_dir+'result_'+file, img_processed )



# run your solution on all test_images and make copies into the test_images directory).

# ## Test on Videos
#
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
#
# We can test our solution on two provided videos:
#
# `solidWhiteRight.mp4`
#
# `solidYellowLeft.mp4`

# In[20]:

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[21]:

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    result = QH_process_image(image)
    return result


# Let's try the one with the solid white lane on the right first ...

# In[22]:
DEBUG_SWITCH = 0

white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[ ]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# **At this point, if you were successful you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform.  Modify your draw_lines function accordingly and try re-running your pipeline.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[ ]:

yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')


# In[ ]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Reflections
#
# Congratulations on finding the lane lines!  As the final step in this project, we would like you to share your thoughts on your lane finding pipeline... specifically, how could you imagine making your algorithm better / more robust?  Where will your current algorithm be likely to fail?
#
# Please add your thoughts below,  and if you're up for making your pipeline more robust, be sure to scroll down and check out the optional challenge video below!
#

# ## Submission
#
# If you're satisfied with your video outputs it's time to submit!
#

# ## Optional Challenge
#
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[ ]:

challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
get_ipython().magic('time challenge_clip.write_videofile(challenge_output, audio=False)')


# In[ ]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))

