from shapedetector import ShapeDetector
import imutils
from cv2 import *
import numpy as np
import time

contour_dist_max = 2500
contour_color = (0, 255, 0)
text_color = (255, 255, 255)
blur_radius = 25
struct_ele = (5, 5)


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape


class ObjectDetection:
    def __init__(self, img):
        self.__img = img
        self.__previous_contours = []

    # get gray image which uses green value as luminosity
    def greenImage(self):
        return split(self.__img)[1]

    # get gray image which uses red value as luminosity
    def redImage(self):
        return split(self.__img)[2]

    # get gray image which uses blue value as luminosity
    def blueImage(self):
        return split(self.__img)[0]

    # get y in yuy2 encoding
    def YUY2(self):
        return split(cvtColor(self.__img, COLOR_RGB2YUV))[0]

    # find contours of objects in binary image
    def find_contour(self, binary, color):
        cont = imutils.grab_contours(findContours(binary, RETR_TREE, CHAIN_APPROX_SIMPLE))
        cont_used = []
        pos = ""
        sd = ShapeDetector()
        for a in range(1, len(cont)):
            c = cont[a]
            # compute the center of the contour
            M = moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            shape = sd.detect(c)
            isSame = False
            for contour in self.__previous_contours:
                contour_dist = (contour[0] - cX) ** 2 + (contour[1] - cY) ** 2
                if contour_dist < contour_dist_max:
                    isSame = True
            if isSame:
                continue
            else:
                self.__previous_contours.append([cX, cY])
                cont_used.append(c)
            # print( shape + "  test shape\n " )
            b, g, r = self.__img[cY, cX]
            drawContours(self.__img, [c], -1, contour_color, 2)
            putText(self.__img, color, (cX - 20, cY - 20),
                    FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            pos += color + "\t"
        return pos

    # use a specific range of colors as mask and transfer image to binary image
    def get_color(self, lower_range, upper_range, color):

        hsv = cv2.cvtColor(self.__img, cv2.COLOR_BGR2HSV)

        # imshow('image', self)
        # waitKey(0)
        imshow("hsv", hsv);

        lower_range = np.array([lower_range])
        upper_range = np.array([upper_range])

        mask = cv2.inRange(hsv, lower_range, upper_range)
        kernel = np.ones(struct_ele, np.uint8)
        opening = morphologyEx(mask, MORPH_OPEN, kernel)
        not_opening = bitwise_not(opening)
        # imshow('image', not_opening)
        # waitKey(0)
        return self.find_contour(not_opening, color)

    # find vrx_target objects
    # output: 'light buoy's color'
    def return_color(self):
        # redimg = object_detection(redImage(self), self)
        # greenimg = object_detection(greenImage(self), self)
        red = self.get_color((0, 200, 50), (3, 255, 255), 'R')
        green = self.get_color((55, 200, 50), (65, 255, 255), 'G')
        blue = self.get_color((115, 200, 50), (125, 255, 255), 'B')
        # white = get_color(self, (0, 0, 40), (3, 3, 50), 'white')
        # other1 = object_detection(greenImage(self), self, 'surmark_46104')

        total = red + green + blue
        return total.strip("\t")

    def color_order(self, order, last_color, recent_color):

        if recent_color != last_color and recent_color != '':
            order.append(recent_color)

        return order


if __name__ == '__main__':

    cap = cv2.VideoCapture('robotX.mp4')
    order = []
    last_color = ''

    ret, frame = cap.read()
    avg = cv2.blur(frame, (4, 4))
    avg_float = np.float32(avg)

    while (cap.isOpened()):

        ret, frame = cap.read()

        # video ends
        if ret == False:
            break

        if cv2.waitKey(16) & 0xFF == ord('q'):
            break
        # time.sleep(5)
        od = ObjectDetection(frame)

        recent_color = od.return_color()

        order = od.color_order(order, last_color, recent_color)

        last_color = recent_color

        cv2.imshow('frame', frame)

    print(order)

    cap.release()
    cv2.destroyAllWindows()
