import cv2
import os
import argparse
from network_model import model
from aux_functions import *
import tsanalyser
tsanalyser.startKeepingTime()



# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

mouse_pts = []


def get_mouse_points(event, x, y, flags, param):
    mouse_pts = [(31, 227), (226, 697), (534, 6), (799, 424), (718, 182), (715, 368), (715, 261)]

    # Used to mark 4 points on the frame zero of the video that will be warped
    # Used to mark 2 points on the frame zero of the video that are 6 feet away
'''    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(image, (x, y), 10, (0, 255, 255), 10)
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        print("Point detected")
        print(mouse_pts)
'''


# Command-line input setup
parser = argparse.ArgumentParser(description="SocialDistancing")
parser.add_argument(
    "--videopath", type=str, default="vid_short.mp4", help="Path to the video file"
)
args = parser.parse_args()

input_video = args.videopath

# Define a DNN model
DNN = model()
# Get video handle
cap = cv2.VideoCapture(input_video)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))

scale_w = 1.2 / 2
scale_h = 4 / 2

SOLID_BACK_COLOR = (41, 41, 41)
# Setuo video writer
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_movie = cv2.VideoWriter("Pedestrian_detect.avi", fourcc, fps, (width, height))
bird_movie = cv2.VideoWriter(
    "Pedestrian_bird.avi", fourcc, fps, (int(width * scale_w), int(height * scale_h))
)
# Initialize necessary variables
frame_num = 0
total_pedestrians_detected = 0
total_six_feet_violations = 0
total_pairs = 0
abs_six_feet_violations = 0
pedestrian_per_sec = 0
sh_index = 1
sc_index = 1
'''
cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)
num_mouse_points = 0
first_frame_display = True
'''

# Process each frame, until end of video
while cap.isOpened():
    frame_num += 1
    ret, frame = cap.read()

    if not ret:
        print("end of the video file...")
        break

    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    print("Processing frame: ", frame_num)
    four_points=[(31, 227), (226, 697), (534, 6), (799, 424), (718, 182), (715, 368), (715, 261)]
    # draw polygon of ROI
    pts = np.array(
        [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32
    )
    cv2.polylines(frame, [pts], True, (0, 255, 255), thickness=4)

    # Detect person and bounding boxes using DNN
    pedestrian_boxes, num_pedestrians = DNN.detect_pedestrians(frame)

    if len(pedestrian_boxes) > 0:
        pedestrian_detect = plot_pedestrian_boxes_on_image(frame, pedestrian_boxes)


    last_h = 75
    text = "# 6ft violations: " + str(int(total_six_feet_violations))
    pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)
    print(text)
    text = "Stay-at-home Index: " + str(np.round(100 * sh_index, 1)) + "%"
    pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)
    print(text)
    if total_pairs != 0:
        sc_index = 1 - abs_six_feet_violations / total_pairs

    text = "Social-distancing Index: " + str(np.round(100 * sc_index, 1)) + "%"
    pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)
    print(text)
    break

time = tsanalyser.getTimeTaken(10)
ram = tsanalyser.getCurrentRSS(10)
print(ram)

print(time)

