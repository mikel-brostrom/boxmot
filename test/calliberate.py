from utils.transform import four_point_transform
import numpy as np
import argparse
import cv2
from PIL import Image, ImageOps
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-c", "--coords", help="comma seperated list of source points")
args = vars(ap.parse_args())
image = cv2.imread("calliberation_image.jpg")
image2 = cv2.imread("struct2.JPG")
close1 = False
pointIndex = 0
pts = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
pts2 = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
AR = (740, 1280)
oppts = np.float32([[0, 0], [AR[1], 0], [0, AR[0]], [AR[1], AR[0]]])

# function to select four points on a image to capture desired region
def draw_circle(event, x, y, flags, param):
    image = param[0]
    pts = param[1]
    global pointIndex
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        pts[pointIndex] = (x, y)
        # print(pointIndex)
        if pointIndex == 3:
            cv2.line(image, pts[0], pts[1], (0, 255, 0), thickness=2)
            cv2.line(image, pts[0], pts[2], (0, 255, 0), thickness=2)
            cv2.line(image, pts[1], pts[3], (0, 255, 0), thickness=2)
            cv2.line(image, pts[2], pts[3], (0, 255, 0), thickness=2)

        pointIndex = pointIndex + 1

def show_window(image, string):
    global pointIndex
    while True:
        # print(pts,pointIndex-1)
        cv2.imshow(string, image)

        if cv2.waitKey(20) & 0xFF == 27:
            break
        if pointIndex == 4:
            pointIndex = 0
            #cv2.destoryWindow(string)
            break

for i in range(2):
    cv2.namedWindow("img2")
    pts = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    pts2 = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]

    cv2.setMouseCallback("img2", draw_circle, param=(image2, pts2))

    show_window(image2, "img2")
    cv2.namedWindow("img")

    cv2.setMouseCallback("img", draw_circle, param=(image, pts))
    show_window(image, "img")
    #W, H 구해야함.
    #height, width, channels = image2.shape
    #background = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    #nparray1 = np.array(pts2[:4], dtype="float32")
    close1 = True
    nparray2 = np.array(pts[:4], dtype="float32")
    W = pts2[3][0] - pts2[0][0]
    H = pts2[3][1] - pts2[0][1]
    nparray1 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
    M = cv2.getPerspectiveTransform(nparray2, nparray1)
    warped = cv2.warpPerspective(image, M, (W, H))
    #print(pts2)
    str_npy = "transformation_matrix" + str(i+1) + ".npy"
    np.save(str_npy, M)
    #temp = cv2.perspectiveTransform(np.array(pts, dtype=np.float32).reshape(1, -1, 2), M)
    #a, b = tuple(temp[0][0])
    #x = (int(a), int(b))
    #cv2.circle(warped, x, 1, (0, 255, 0), -1)
    #            img, center,          radius, color, thickness, (lineType, shift)
    # show the original and warped images
    str_coor = "coor" + str(i+1) + ".txt"
    f = open(str_coor, 'w')
    f.write(str(pts2[0][0]) + ' ' + str(pts2[0][1]))
    f.close()
    cv2.imshow("Original", image)
    #cv2.imshow("Warped", warped) #Uncommment if you want to see the transformed image
    cv2.waitKey(0)
    cv2.destroyAllWindows()
