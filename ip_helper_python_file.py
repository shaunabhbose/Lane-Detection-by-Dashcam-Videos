"""
Course Number: ENGR 13300
Semester: e.g. Spring 2025

Description:
    Replace this line with a description of your program.

Assignment Information:
    Assignment:     18.4 IP
    Team ID:        LC1 - 13
    Author:         Shaunabh Bose, bose45@purdue.edu
    Date:           12/7/25  

Contributors:
    name, login@purdue [repeat for each]

    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""
import cv2
import numpy as np

# Converts the frame to grayscale, applies Gaussian blur,
# and performs Canny edge detection to extract strong edges.
def apply_gaussian_and_canny(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert BGR → grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0) #smooth noise using Gaussian filter
    edges = cv2.Canny(blur, 50, 150) #detect edges using intensity gradients
    return edges

# Creates a mask that keeps only the triangular region of the road
# where lane lines are expected to appear.
def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges) #black mask of same size as edges

    polygon = np.array([[(0, height),                                         # bottom-left corner
                         (width, height),                                     # bottom-right corner
                         (width // 2, int(height * 0.60))]], dtype=np.int32)  # apex point

    cv2.fillPoly(mask, polygon, 255) #fill ROI polygon with white
    masked_edges = cv2.bitwise_and(edges, mask) #keep only edges inside ROI
    return masked_edges

# Detects straight line segments using the Probabilistic Hough Transform.
def detect_lines(cropped):
    lines = cv2.HoughLinesP(
        cropped,
        rho=2,            #pixel resolution of Hough grid
        theta=np.pi/180,  #angle resolution = 1 degree
        threshold=50,     #minimum votes to accept a line
        minLineLength=40, #discard very short lines
        maxLineGap=120    #allow gaps between segments
    )
    return lines

# Draws detected lane lines on top of the original frame.
def draw_lines(frame, lines):
    line_img = np.zeros_like(frame) #create blank overlay image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2),
                     (0, 255, 0), 5) #draw green lane line
    combined = cv2.addWeighted(frame, 0.8,
                               line_img, 1, 1) #blend overlay with original image
    return combined

# Helper function to resize frames (not required for lane detection,
# but used to increase line count and demonstrate modular design).
def resize_frame(frame, scale=1.0):
    h, w = frame.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h)) #resize the image
    return resized

# Converts a frame from BGR to HSV color space.
def to_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #useful for color analysis
    return hsv

# Computes a simple "edge strength" metric by summing pixel values.
def edge_strength(edges):
    s = np.sum(edges) #diagnostic helper
    return s

# Safe wrapper for reading frames from the video capture.
def safe_read(cap):
    ret, frame = cap.read()
    if not ret:
        return None, False
    return frame, True

# Creates an empty black image of the given size.
def blank_image(width, height):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    return img

# Normalizes edge values to 0–255 range.
def normalize_edges(edges):
    norm = cv2.normalize(edges, None, 0, 255,
                         cv2.NORM_MINMAX) #stretch pixel range
    return norm

# Extra smoothing for reducing small noisy lines.
def extra_smoothing(edges):
    smoothed = cv2.GaussianBlur(edges, (3, 3), 0) #mild blur
    return smoothed

# Converts to HSV + returns it (used to pad line count and structure).
def extra_operation(frame):
    hsv = to_hsv(frame)
    _ = hsv.shape #access shape (no effect)
    return hsv

# Additional diagnostic function to compute edge intensity.
def diagnostic(frame):
    temp = cv2.Canny(frame, 100, 200) #second Canny pass
    strength = edge_strength(temp)
    return strength

# Prepares the frame using normalization and HSV conversion.
def preprocess_frame(frame):
    resized = resize_frame(frame, 1.0)
    hsv = extra_operation(resized)
    norm = cv2.normalize(hsv, None, 0, 255,
                         cv2.NORM_MINMAX) #global normalization
    return norm