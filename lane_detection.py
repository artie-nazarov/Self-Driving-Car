import cv2
import numpy as np
import math

# Converting input images to HSV format
def convert_to_HSV(frame):
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  cv2.imshow("HSV", hsv)
  return hsv

# Detect white edges (road lines)
def detect_edges(frame):
# white edges
  sensitivity = 100
  lower_white = np.array([0,0,255-sensitivity])
  upper_white = np.array([255,sensitivity,255])

  mask = cv2.inRange(frame, lower_white, upper_white)

  # detect edges
  edges = cv2.Canny(mask, 50, 100)
  cv2.imshow("edges", edges)
  return edges

# Select the region of interest
def region_of_interest(edges):
  height, width = edges.shape
  mask = np.zeros_like(edges)

  # only focus lower half of the screen
  polygon = np.array([[(0, height),
                       (0, height/2),
                       (width, height/2),
                       (width, height),
                       ]], np.int32)

  cv2.fillPoly(mask, polygon, 255)
  cropped_edges = cv2.bitwise_and(edges, mask)
  cv2.imshow("roi", cropped_edges)
  return cropped_edges

# Detect line segments
def detect_line_segments(cropped_edges):
  rho = 1 # distance precision in pixels
  theta = np.pi / 180 # ~ 1 degree
  min_threshold = 10 # min votes for shape to be considered a line
  line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold,
                                  np.array([]), minLineLength=5, maxLineGap=0)
  return line_segments

# Helper function that will return bounded coordinates of the lane lines
def make_points(frame, line):
  height, width, _ = frame.shape
  slope, intercept = line
  y1 = height  # bottom of the frame
  y2 = int(y1 / 2)  # make points from middle of the frame down

  # if the slope = 0 give it a small value (avoid dividing by 0)
  if slope == 0:
    slope = 0.1

  x1 = int((y1 - intercept) / slope)
  x2 = int((y2 - intercept) / slope)

  return [[x1, y1, x2, y2]]

# Compute average slope intercept
def average_slope_intercept(frame, line_segments):
  lane_lines = []

  if line_segments is None:
    print("no line segment was detected")
    return lane_lines

  height, width, _ = frame.shape
  left_fit = []
  right_fit = []
  boundary = 1/3

  left_region_boundary = width * (1 - boundary)
  right_region_boundary = width * boundary

  for line_segment in line_segments:
    for x1, y1, x2, y2 in line_segment:
      if x1 == x2:
        print("skipping vertical lines (slope = infinity)")
        continue

      fit = np.polyfit((x1, x2), (y1, y2), 1)
      slope = (y2 - y1) / (x2 - x1)
      intercept = y1 - (slope * x1)

      if slope < 0:
        if x1 < left_region_boundary and x2 < left_region_boundary:
          left_fit.append((slope, intercept))
      else:
        if x1 > right_region_boundary and x2 > right_region_boundary:
          right_fit.append((slope, intercept))

  left_fit_average = np.average(left_fit, axis=0)
  if len(left_fit) > 0:
    lane_lines.append(make_points(frame, left_fit_average))

  right_fit_average = np.average(right_fit, axis=0)
  if len(right_fit) > 0:
    lane_lines.append(make_points(frame, right_fit_average))

  return lane_lines

# Display the lane lines on the frames
def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6):  # line color (B,G,R)
  line_image = np.zeros_like(frame)

  if lines is not None:
    for line in lines:
      for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)

  # used for combining images together
  line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
  return line_image


## Get steering angles
def get_steering_angle(frame, lane_lines):
  height, width, _ = frame.shape

  if len(lane_lines) == 2: # 2 lane lines are detected
    _, _, left_x2, _ = lane_lines[0][0] # extract left x2 from lane_lines array
    _, _, right_x2, _ = lane_lines[1][0] # extract right x2 from lane lines array

    mid = int(width / 2)
    x_offset = (left_x2 + right_x2) / 2 - mid
    y_offset = int(height / 2)

  elif len(lane_lines) == 1:  # if only one line is detected
    x1, _, x2, _ = lane_lines[0][0]
    x_offset = x2 - x1
    y_offset = int(height / 2)

  elif len(lane_lines) == 0:  # if no line is detected
    x_offset = 0
    y_offset = int(height / 2)

  angle_to_mid_radian = math.atan(x_offset / y_offset)
  angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
  steering_angle = angle_to_mid_deg + 90

  return steering_angle


# Display Heading Line
def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5):

  heading_image = np.zeros_like(frame)
  height, width, _ = frame.shape

  steering_angle_radian = steering_angle / 180.0 * math.pi
  x1 = int(width / 2)
  y1 = height
  x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
  y2 = int(height / 2)

  cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)

  heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

  return heading_image


# main
def main():
  video = cv2.VideoCapture(0)
  video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
  video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

  while True:
    ret, frame = video.read()
    frame = cv2.flip(frame, -1)

    # Calling the functions
    hsv = convert_to_HSV(frame)
    edges = detect_edges(hsv)
    roi = region_of_interest(edges)
    line_segments = detect_line_segments(roi)
    lane_lines = average_slope_intercept(frame, line_segments)
    lane_lines_image = display_lines(frame, lane_lines)
    steering_angle = get_steering_angle(frame, lane_lines)
    heading_image = display_heading_line(lane_lines_image, steering_angle)

    key = cv2.waitKey(1)
    if key == 27:
      break

  video.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()