import cv2
import numpy as np
import scipy.ndimage

def preprocess(raw_data) :
    return smallPreprocess(raw_data)

def erase_circles(img, circles):
    circles = circles[0]  # hough circles returns a nested list for some reason
    for circle in circles:
        center = (round(circle[0]),round(circle[1]))
        img = cv2.circle(img, center, radius=round(circle[2]), color=255, thickness=1)  # erase circle by making it white (to match the image background)
    return img

def detect_and_remove_circles(img):
    hough_circle_locations = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=5, minRadius=0, maxRadius=2)
    if hough_circle_locations is not None:
        img = erase_circles(img, hough_circle_locations)
    return img

def smallPreprocess(raw_data):
    img = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
    image = np.array(img) / 255.0
    (c, h, w) = image.shape
    image = image.reshape([-1, c, h, w])
    return image

def pyPreprocess(raw_data):
    img = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
    image = np.array(img, dtype=np.float32) / 255.0
    (c, h, w) = image.shape
    image = image.reshape([-1, c, h, w])
    return image

def smallPreprocess(raw_data):
    img = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY) # Gray Image
    img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)  # dilate image to initial stage (erode works similar to dilate because we thresholded the image the opposite way)
    img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)  # erode just a bit to polish fine details
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # Back to Colour for channel
    image = np.array(img) / 255.0
    (c, h, w) = image.shape
    image = image.reshape([-1, c, h, w])
    return image

def pyMidPreprocess(raw_data):
    img = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY) # Gray Image
    img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)  # dilate image to initial stage (erode works similar to dilate because we thresholded the image the opposite way)
    img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)  # erode just a bit to polish fine details
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # Back to Colour for channel
    # image = np.array(img) / 255.0
    image = np.array(img, dtype=np.float32) / 255.0
    (c, h, w) = image.shape
    image = image.reshape([-1, c, h, w])
    return image

def bigPreprocess(raw_data) :
    #   Back to Black
    img = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY)
    # run some basic tests to get rid of easy-to-remove noise -- first pass
    img = ~img  # white letters, black background
    img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)  # weaken circle noise and line noise
    img = ~img  # black letters, white background
    img = scipy.ndimage.median_filter(img, (5, 1))  # remove line noise
    img = scipy.ndimage.median_filter(img, (1, 3))  # weaken circle noise
    img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)  # dilate image to initial stage (erode works similar to dilate because we thresholded the image the opposite way)
    img = scipy.ndimage.median_filter(img, (3, 3))  # remove any final 'weak' noise that might be present (line or circle)
    # detect any remaining circle noise
    img = detect_and_remove_circles(img)  # after dilation, if concrete circles exist, use hough transform to remove them
    # eradicate any final noise that wasn't removed previously -- second pass
    img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)  # actually performs erosion
    img = scipy.ndimage.median_filter(img, (5, 1))  # finally completely remove any extra noise that remains
    img = cv2.erode(img, np.ones((3, 3), np.uint8), iterations=2)  # dilate image to make it look like the original
    img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)  # erode just a bit to polish fine details
    #Edge Detection
    # img = cv2.Canny(img, 100, 200)
    # Back to Colour
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Format - (different for pi)
    image = np.array(img) / 255.0
    (c, h, w) = image.shape
    return image.reshape([-1, c, h, w])