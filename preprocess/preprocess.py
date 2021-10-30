import cv2
import numpy as np
import scipy.ndimage

def preprocess(raw_data) :
    return smallPreprocess(raw_data)

def lose_circles(i, cs):
    cs = cs[0]
    for c in cs:
        i = cv2.circle(i, (round(c[0]),round(c[1])), radius=round(c[2]), color=255, thickness=1)
    return i

def circles(i):
    cs = cv2.HoughCircles(i, method=cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=5, minRadius=0, maxRadius=2)
    if cs is not None:
        i = lose_circles(i, cs)
    return i

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

def bigPreprocess(raw_data) :
    #   Back to Black
    img = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY)
    # First Pass
    img = ~img  # invert
    img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)  # weaken noise
    img = ~img  # re-invert
    img = scipy.ndimage.median_filter(img, (5, 1))  # target lines
    img = scipy.ndimage.median_filter(img, (1, 3))  # target circles
    img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)
    img = scipy.ndimage.median_filter(img, (3, 3))  # target weak noise
    img = circles(img)  # Use Hough Transform on remaining circles
    # Last Pass
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
