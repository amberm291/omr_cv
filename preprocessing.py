import cv2
import numpy as np 
import os
from matplotlib import pyplot as plt
from scipy import ndimage

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

    def remove_staff_lines(self, staff_pixels, img):
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

        diff_list = map(lambda x:self.staff_lines[x] - self.staff_lines[x-1],range(1,5))
        self.diff = float(sum(diff_list))/len(diff_list)
        self.diff = int(round(self.diff))

        lines = filter(lambda x:x%5==0,range(len(self.staff_lines)))
        for i in lines:
            staff_pixels.extend(range(self.staff_lines[i]-self.diff-1,self.staff_lines[i]-self.diff+1))
            staff_pixels.extend(range(self.staff_lines[i]-2*self.diff-1,self.staff_lines[i]-2*self.diff+1))

        lines = filter(lambda x:x%5==4,range(len(self.staff_lines)))
        for i in lines:
            staff_pixels.extend(range(self.staff_lines[i]+self.diff-1,self.staff_lines[i]+self.diff+1))
            staff_pixels.extend(range(self.staff_lines[i]+2*self.diff-1,self.staff_lines[i]+2*self.diff+1))

        for pixel in staff_pixels:
            img[pixel,:] = 255

    def detect_staff_pixels(self, img):
        staff_pixels = []
        height, width = img.shape
        ref_img = np.copy(img)
        for i in xrange(height):
            num_zero = width - np.count_nonzero(img[i,:])
            if num_zero > width/2: 
                staff_pixels.append(i)
        
        return staff_pixels

    def close_gaps(self,img):
        kernel = np.ones((3,3),np.uint8)
        ref_img = cv2.erode(img,kernel)
        kernel = np.ones((3,3),np.uint8)
        ref_img = cv2.dilate(ref_img,kernel)
        return ref_img

    def run(self, fname):
        img = self.open_file(fname)
        img = self.binarize_file(img)
        staff_pixels = self.detect_staff_pixels(img)
        self.remove_staff_lines(staff_pixels, img)
        img = self.close_gaps(img)
        cv2.imwrite(self.data_dir + "staff_removed.png",img)

if __name__=="__main__":
    data_dir = "/Users/ambermadvariya/data/omr_cv/"
    fname = "sample.png"
    pre_inst = preProcess(data_dir)
    pre_inst.run(fname)
    