import cv2
import numpy as np 
import os
from matplotlib import pyplot as plt

class preProcess:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def open_file(self, fname):
        if not os.path.exists(self.data_dir + fname):
            raise IOError("File doesn't exist.")
        return cv2.imread(self.data_dir + fname,0)

    def binarize_file(self, img):
        ret, thresh_img = cv2.threshold(img,160,255,cv2.THRESH_BINARY)
        return thresh_img

    def detect_staff_pixels(self, img):
        staff_pixels = []
        height, width = img.shape
        ref_img = np.copy(img)
        for i in xrange(height):
            num_zero = width - np.count_nonzero(img[i,:])
            if num_zero > width/2: 
                staff_pixels.append(i)
                ref_img[i,:] = 255

        kernel = np.ones((3,3),np.uint8)
        ref_img = cv2.erode(ref_img,kernel)
        kernel = np.ones((3,3),np.uint8)
        ref_img = cv2.dilate(ref_img,kernel)
        self.staff_lines = []
        group_staff_lines = []
        for i in xrange(1,len(staff_pixels)):
            if (staff_pixels[i] - staff_pixels[i-1]) < 5:
                group_staff_lines.append(staff_pixels[i-1])
            else:
                group_staff_lines.append(staff_pixels[i-1])
                self.staff_lines.append(sum(group_staff_lines)/len(group_staff_lines))
                group_staff_lines = []
        
        if len(group_staff_lines) != 0:
            self.staff_lines.append(sum(group_staff_lines)/len(group_staff_lines))

        cv2.imwrite(self.data_dir + "staff_freq.png",ref_img)

    def run(self, fname):
        img = self.open_file(fname)
        img = self.binarize_file(img)
        self.detect_staff_pixels(img)

if __name__=="__main__":
    data_dir = "/Users/ambermadvariya/data/omr_cv/"
    fname = "sample.png"
    pre_inst = preProcess(data_dir)
    pre_inst.run(fname)
    