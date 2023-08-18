import cv2
import numpy as np
from PIL import Image
from readbbtxt_inv import readbbtxt

datafolder = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/data_fin/'
datafile = 'pixel_position_invis_new.txt'

data = readbbtxt(datafolder + datafile)

video_filename = '51007_sessie1_taskrobotEngagement'

vis_root = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/frames/Saved_frame/fig_vis_l2cs/'
# inv_root = '/home/linlincheng/Gaze_estimation/gaze-automated-annotation/bb/frames/Saved_frame/fig_invis_l2cs/'
inv_root = '/home/linlincheng/Gaze_estimation/L2CS-Net/input/fig_invis_l2cs/'
filename = '00050.jpg'
image = inv_root + filename

img = cv2.imread(image)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# (210, 480), (450, 640), (455, 410), (925, 840), (780, 480), (1000, 660)
tl_tablet_x = 210
br_tablet_x = 450
tl_tablet_y = 480
br_tablet_y = 640
# roi = gray[y1:y2, x1:x2]

tl_pp_y = 480
br_pp_y = 660
tl_pp_x = 780
br_pp_x = 1000

tl_robot_y = 410
br_robot_y = 840
tl_robot_x = 455
br_robot_x = 925

roi_tablet = gray[tl_tablet_y:br_tablet_y, tl_tablet_x:br_tablet_x]
roi_pp = gray[tl_pp_y:br_pp_y, tl_pp_x:br_pp_x]
roi_robot = gray[tl_robot_y:br_robot_y, tl_robot_x:br_robot_x]

edges_tablet = cv2.Canny(roi_tablet, threshold1=50, threshold2=150)
edges_pp = cv2.Canny(roi_pp, threshold1=50, threshold2=150)
edges_robot = cv2.Canny(roi_robot, threshold1=50, threshold2=150)

lines_tablet = cv2.HoughLinesP(edges_tablet, rho=1, theta=np.pi / 180, threshold=100, minLineLength=5, maxLineGap=10)
lines_pp = cv2.HoughLinesP(edges_pp, rho=1, theta=np.pi / 180, threshold=100, minLineLength=5, maxLineGap=10)
lines_robot = cv2.HoughLinesP(edges_robot, rho=1, theta=np.pi / 180, threshold=100, minLineLength=5, maxLineGap=10)

# roi_tablet = Image.fromarray(roi_tablet)
# roi_tablet.show()
# roi_pp = Image.fromarray(roi_pp)
# roi_pp.show()
# roi_robot = Image.fromarray(roi_robot)
# roi_robot.show()
#
# print(edges_tablet)
# print(edges_pp)
# print(edges_robot)
#
print(f"lines_tablet: {lines_tablet}")
print(f"lines_pp: {lines_pp}")
print(f"lines_robot: {lines_robot}")

# edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# edges = cv2.Canny(roi, threshold1=50, threshold2=150)

# lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
# lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=5, maxLineGap=10)

rec = data[data['file'] == filename].iloc[0]

x_left_tab = int(rec['tl_tablet_x'] * w)
x_right_tab = int(rec['br_tablet_x'] * w)
y_bottom_tab = int(rec['br_tablet_y'] * h)
y_top_tab = int(rec['tl_tablet_y'] * h)

x_left_pp = int(rec['tl_pp_x'] * w)
x_right_pp = int(rec['br_pp_x'] * w)
y_bottom_pp = int(rec['br_pp_y'] * h)
y_top_pp = int(rec['tl_pp_y'] * h)

x_left_rbt = int(rec['tl_robot_x'] * w)
x_right_rbt = int(rec['br_robot_x'] * w)
y_bottom_rbt = int(rec['br_robot_y'] * h)
y_top_rbt = int(rec['tl_robot_y'] * h)

box_corners_tablet = [(x_left_tab, y_bottom_tab), (x_right_tab, y_bottom_tab), (x_right_tab, y_top_tab),
                      (x_left_tab, y_top_tab)]
box_corners_pp = [(x_left_pp, y_bottom_pp), (x_right_pp, y_bottom_pp), (x_right_pp, y_top_pp),
                  (x_left_pp, y_top_pp)]
box_corners_robot = [(x_left_rbt, y_bottom_rbt), (x_right_rbt, y_bottom_rbt), (x_right_rbt, y_top_rbt),
                     (x_left_rbt, y_top_rbt)]

def draw_line(a, b, c, d, image_in, pitchyaw, thickness=2, color=(255, 255, 0), sclae=2.0):
    """Draw lines and gaze on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    # length = 852/2
    length = w * 999
    pos = (int(a + c / 2.0), int(b + d / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])

    # gazex = round(pos[0] + dx)
    # gazey = round(pos[1] + dy)
    # gaze_destination = tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int))

    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                    tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                    thickness, cv2.LINE_AA, tipLength=0.18)
    cv2.arrowedLine(image_out, )
    return image_out

def classify_line_box_intersection(a, b, c, d, image_in, pitch_pred, jaw_pred):
    intersection_points_tablet = []
    intersection_points_pp = []
    intersection_points_robot = []
    # box_corners = [(1, 1), (3, 1), (3, 3), (1, 3)]

    if pitch_pred == NOFACE and jaw_pred == NOFACE:
        return 4

    (h, w) = image_in.shape[:2]
    length = w * 999
    pos = (int(a + c / 2.0), int(b + d / 2.0))
    dx = -length * np.sin(pitch_pred) * np.cos(jaw_pred)
    dy = -length * np.sin(jaw_pred)
    gazex = round(pos[0] + dx)
    gazey = round(pos[1] + dy)

    # print(f"x: {pos[0]}, y: {pos[1]}")
    # print(f"old gazex: {gazex}, old gazey: {gazey}")

    try:
        gaze_slope = round((gazey - pos[1]) / (gazex - pos[0]))
        # print(f"gaze slope: {gaze_slope}")
    except ZeroDivisionError:
        gaze_slope = 1.0
    gaze_intercept = round(pos[1] - gaze_slope * pos[0])
    # print(f"gaze intercept: {gaze_intercept}")

    # Check if gaze vector leaves the image
    if gazex < 0 or gazex > w or gazey < 0 or gazey > h:
        # Do line intersection to find at which point the gaze vector line
        # leaves the image.
        image_edges = [(0, 0), (w, 0), (w, h), (0, h)]

        # y3 = gazey if 0 <= gazey <= h else 0 if gazey < 0 else h
        # y4 = y3

        x1, y1 = round(pos[0]), round(pos[1])
        x2, y2 = gazex, gazey

        for edge in [1, 2, 3]:
            x3, y3 = image_edges[edge]
            x4, y4 = image_edges[(edge + 1) % 4]  # Next corner (wraps around to the first corner)

            gazex, gazey = line_intersect(x1, y1, x2, y2, x3, y3, x4, y4)

            if min(x1, x2) <= gazex <= max(x1, x2) and min(y1, y2) <= gazey <= max(y1, y2):
                # print(gazex, gazey)
                break

        # print(f"new gazex: {gazex}, new gazey: {gazey}")

        # # Calculate the y-values at the x-coordinates of the line segment's endpoints
        # y1_intersect = gaze_slope * x1 + gaze_intercept
        # y2_intersect = gaze_slope * x2 + gaze_intercept
        #
        # # Check if the y-values lie within the range of the line segment
        # if (y1_intersect * y2_intersect <= 0) or (y1_intersect == 0) or (y2_intersect == 0):
        #     return True
        # else:
        #     return False

    rec = data[data['file'] == filename].iloc[0]

    x_left_tab = int(rec['tl_tablet_x'] * w)
    x_right_tab = int(rec['br_tablet_x'] * w)
    y_bottom_tab = int(rec['br_tablet_y'] * h)
    y_top_tab = int(rec['tl_tablet_y'] * h)

    x_left_pp = int(rec['tl_pp_x'] * w)
    x_right_pp = int(rec['br_pp_x'] * w)
    y_bottom_pp = int(rec['br_pp_y'] * h)
    y_top_pp = int(rec['tl_pp_y'] * h)

    x_left_rbt = int(rec['tl_robot_x'] * w)
    x_right_rbt = int(rec['br_robot_x'] * w)
    y_bottom_rbt = int(rec['br_robot_y'] * h)
    y_top_rbt = int(rec['tl_robot_y'] * h)

    box_corners_tablet = [(x_left_tab, y_bottom_tab), (x_right_tab, y_bottom_tab), (x_right_tab, y_top_tab),
                          (x_left_tab, y_top_tab)]
    box_corners_pp = [(x_left_pp, y_bottom_pp), (x_right_pp, y_bottom_pp), (x_right_pp, y_top_pp),
                      (x_left_pp, y_top_pp)]
    box_corners_robot = [(x_left_rbt, y_bottom_rbt), (x_right_rbt, y_bottom_rbt), (x_right_rbt, y_top_rbt),
                         (x_left_rbt, y_top_rbt)]

    # print(box_corners_tablet)
    # print(box_corners_pp)
    # print(box_corners_robot)

    roi_tablet = image_in[y_top_tab:y_bottom_tab, x_left_tab:x_right_tab]
    roi_pp = image_in[y_top_pp:y_bottom_pp, x_left_pp:x_right_pp]
    roi_robot = image_in[y_top_rbt:y_bottom_rbt, x_left_rbt:x_right_rbt]

    # roi_tablet = Image.fromarray(roi_tablet)
    # roi_tablet.show()
    # roi_pp = Image.fromarray(roi_pp)
    # roi_pp.show()
    # roi_robot = Image.fromarray(roi_robot)
    # roi_robot.show()

    # check if there is any the intersection point in tablet box
    for i in range(4):
        x1_t, y1_t = box_corners_tablet[i]
        x2_t, y2_t = box_corners_tablet[(i + 1) % 4]  # Next corner (wraps around to the first corner)
        # print((x1_t, y1_t))
        # print((x2_t, y2_t))
        # Check if the line and line segment are parallel
        try:
            # tablet_edge_slope = round((y2_t - y1_t) / (x2_t - x1_t))
            tablet_edge_slope = abs((y2_t - y1_t) / (x2_t - x1_t))
            # print(f"tablet edge slope: {tablet_edge_slope}")
        except ZeroDivisionError:
            tablet_edge_slope = 1.0
            # print(f"tablet edge slope: {tablet_edge_slope}")

        if gaze_slope == tablet_edge_slope:
            continue

        # Calculate the intersection point
        # try:
        intersection_x_t = int((gaze_intercept - y1_t + tablet_edge_slope * x1_t) / (tablet_edge_slope - gaze_slope))
        # intersection_x_t = (gaze_intercept - y1_t + tablet_edge_slope * x1_t) / (tablet_edge_slope - gaze_slope)
        # print(f"int x: {intersection_x_t}")
        # except ZeroDivisionError:
        #     intersection_x_t = (gaze_intercept - y1_t + tablet_edge_slope * x1_t) / (tablet_edge_slope - gaze_slope)
        #     print("DZ error: tablet")
        intersection_y_t = int(gaze_slope * intersection_x_t + gaze_intercept)
        # intersection_y_t = gaze_slope * intersection_x_t + gaze_intercept
        # print(f"int y: {intersection_y_t}")

        # Check if the intersection point lies within the range of the line segment
        if min(x1_t, x2_t) <= abs(intersection_x_t) <= max(x1_t, x2_t) and min(y1_t, y2_t) <= abs(
                intersection_y_t) <= max(y1_t,
                                         y2_t):
            intersection_points_tablet.append((intersection_x_t, intersection_y_t))
        # else:
        #     print(f"there is no intersection with tablet line {i}")
        if intersection_points_tablet:
            print(f"tablet: {intersection_points_tablet}")
    # return intersection_points_tablet

    # check if there is any the intersection point in pp box
    for i in range(4):
        x1_p, y1_p = box_corners_pp[i]
        x2_p, y2_p = box_corners_pp[(i + 1) % 4]  # Next corner (wraps around to the first corner)

        # Check if the line and line segment are parallel
        try:
            # pp_edge_slope = round((y2_p - y1_p) / (x2_p - x1_p))
            pp_edge_slope = (y2_p - y1_p) / (x2_p - x1_p)
        except ZeroDivisionError:
            pp_edge_slope = 1.0

        if gaze_slope == pp_edge_slope:
            continue

        # Calculate the intersection point
        # try:
        intersection_x_p = int((gaze_intercept - y1_p + pp_edge_slope * x1_p) / (pp_edge_slope - gaze_slope))
        # intersection_x_p = (gaze_intercept - y1_t + tablet_edge_slope * x1_t) / (tablet_edge_slope - gaze_slope)
        # except ZeroDivisionError:
        #     intersection_x_p = 1
        intersection_y_p = int(gaze_slope * intersection_x_p + gaze_intercept)
        # intersection_y_p = gaze_slope * intersection_x_p + gaze_intercept

        # Check if the intersection point lies within the range of the line segment
        if min(x1_p, x2_p) <= abs(intersection_x_p) <= max(x1_p, x2_p) and min(y1_p, y2_p) <= abs(
                intersection_y_p) <= max(y1_p,
                                         y2_p):
            intersection_points_pp.append((intersection_x_p, intersection_y_p))
        # else:
        #     print(f"there is not intersection with pp line {i}")
        if intersection_points_pp:
            print(f"pp: {intersection_points_pp}")
    # return intersection_points_tablet

    # check if there is any the intersection point in robot box
    for i in range(4):
        x1_r, y1_r = box_corners_robot[i]
        x2_r, y2_r = box_corners_robot[(i + 1) % 4]  # Next corner (wraps around to the first corner)

        # Check if the line and line segment are parallel
        try:
            # robot_edge_slope = round((y2_r - y1_r) / (x2_r - x1_r))
            robot_edge_slope = (y2_r - y1_r) / (x2_r - x1_r)
        except ZeroDivisionError:
            robot_edge_slope = 1.0

        if gaze_slope == robot_edge_slope:
            continue

        # Calculate the intersection point
        # try:
        intersection_x_r = int((gaze_intercept - y1_r + robot_edge_slope * x1_r) / (robot_edge_slope - gaze_slope))
        # intersection_x_r = (gaze_intercept - y1_r + robot_edge_slope * x1_r) / (robot_edge_slope - gaze_slope)
        # except ZeroDivisionError:
        #     intersection_x_r = 1
        intersection_y_r = int(gaze_slope * intersection_x_r + gaze_intercept)
        # intersection_y_r = gaze_slope * intersection_x_r + gaze_intercept

        # Check if the intersection point lies within the range of the line segment
        if min(x1_r, x2_r) <= abs(intersection_x_r) <= max(x1_r, x2_r) and min(y1_r, y2_r) <= abs(
                intersection_y_r) <= max(y1_r, y2_r):
            intersection_points_robot.append((intersection_x_r, intersection_y_r))
        # else:
        #     print(f"there is not intersection with robot line {i}")
        if intersection_points_robot:
            print(f"robot: {intersection_points_robot}")
    # return intersection_points_tablet
    if intersection_points_tablet:  # return 2 if list of intersection points is not empty
        return 2
    elif intersection_points_pp:
        return 0
    elif intersection_points_robot:
        return 1
    else:
        return 3

























"""
if lines_tablet is not None:
    for line in lines_tablet:
        x2, y2, x1, y1 = line[0]
        line_slope = abs((y2-y1)/(x2-x1))
        print(f"tablet: {line_slope}")
        # line_thickness = abs(x2 - x1)  # calculate the line thickness
        # if line_thickness == 2:
        #     print("tablet")
        #     break
        # else:
        #     print(f"tablet: {abs(x2-x1)}")
elif lines_pp is not None:
    for line in lines_pp:
        x1, y1, x2, y2 = line[0]
        line_slope = abs((y2-y1)/(x2-x1))
        print(f"pp: {line_slope}")
        # line_thickness = abs(x2 - x1)  # calculate the line thickness
        # if line_thickness == 2:
        #     print("pp")
        #     break
        # else:
        #     print(f"pp: {abs(x2-x1)}")
elif lines_robot is not None:
    for line in lines_robot:
        x1, y1, x2, y2 = line[0]
        line_slope = abs((y2-y1)/(x2-x1))
        print(f"robot: {line_slope}")
        # line_thickness = abs(y2 - y1)  # calculate the line thickness
        # if line_thickness == 2:
        #     print("robot")
        #     break
        # else:
        #     print(f"robot: {abs(y2 - y1)}")
else:
    print(f"else")

# if lines is not None:
#     print("A line is detected.")
# else:
#     print("No line detected.")"""
