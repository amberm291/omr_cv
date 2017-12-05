import cv2
import numpy as np 
import os
from matplotlib import pyplot as plt
from scipy import ndimage
import Queue

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
        kernel = np.ones((4,4),np.uint8)
        ref_img = cv2.erode(img,kernel)
        kernel = np.ones((4,4),np.uint8)
        ref_img = cv2.dilate(ref_img,kernel)
        return ref_img

    def find_connected_components(self, img):
        height, width = img.shape
        visited = np.zeros((height,width))
        queue = Queue.Queue()
        connected_components = []
        flag = True
        for i in xrange(height):
            for j in xrange(width):
                if visited[i][j] != 0: continue
                if img[i][j] != 0: 
                    visited[i][j] = 1
                    continue
                component = []
                queue.put((i,j))
                component.append((i,j))
                while not queue.empty():
                    row, col = queue.get()
                    if row > 0 and col > 0 and visited[row-1][col-1] == 0 and img[row-1][col-1] == 0:
                        queue.put((row-1,col-1))
                        visited[row-1][col-1] = 1
                        component.append((row-1,col-1))
                    if row > 0 and visited[row-1][col] == 0 and img[row-1][col] == 0:
                        queue.put((row-1,col))
                        visited[row-1][col] = 1
                        component.append((row-1,col))
                    if row > 0 and col < (width-1) and visited[row-1][col+1] == 0 and img[row-1][col+1] == 0:
                        queue.put((row-1,col+1))
                        visited[row-1][col+1] = 1
                        component.append((row-1,col+1))
                    if col > 0 and visited[row][col-1] == 0 and img[row][col-1] == 0:
                        queue.put((row,col-1))
                        visited[row][col-1] = 1
                        component.append((row,col-1))
                    if col < (width-1) and visited[row][col+1] == 0 and img[row][col+1] == 0:
                        queue.put((row,col+1))
                        visited[row][col+1] = 1
                        component.append((row,col+1))
                    if row < (height-1) and col > 0 and visited[row+1][col-1] == 0 and img[row+1][col-1] == 0:
                        queue.put((row+1,col-1))
                        visited[row+1][col-1] = 1
                        component.append((row+1,col-1))
                    if row < (height-1) and visited[row+1][col] == 0 and img[row+1][col] == 0:
                        queue.put((row+1,col))
                        visited[row+1][col] = 1
                        component.append((row+1,col))
                    if row < (height-1) and col < (width-1) and visited[row+1][col+1] == 0 and img[row+1][col+1] == 0:
                        queue.put((row+1,col+1))
                        visited[row+1][col+1] = 1
                        component.append((row+1,col+1))
                row_index = map(lambda x:x[0],component)
                col_index = map(lambda x:x[1],component)
                top = min(row_index)
                bottom = max(row_index)
                left = min(col_index)
                right = max(col_index)
                connected_components.append((top,left,bottom,right))

        return connected_components

    def mark_components(self, img):
        connected_components = self.find_connected_components(img)
        draw_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        for component in connected_components:
            top, left, bottom, right = component
            cv2.rectangle(draw_img,(left,top),(right,bottom),(0,255,0),3)
        cv2.imwrite(self.data_dir + "components.png",draw_img)
            
    def detect_segments(self, img):
        img = 255 - img
        labels, numLabels = ndimage.label(img)
        fragments = ndimage.find_objects(labels)
        draw_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        flag = True
        for slices in fragments:
            top = slices[0].start
            bottom = slices[0].stop
            left = slices[1].start
            right = slices[1].stop
            cv2.rectangle(draw_img,(left,top),(right,bottom),(0,255,0),3)
            if flag:
                print img[top:bottom,left:right]
                flag = False
        cv2.imwrite(self.data_dir + "components.png",draw_img)

    def run(self, fname):
        img = self.open_file(fname)
        img = self.binarize_file(img)
        staff_pixels = self.detect_staff_pixels(img)
        self.remove_staff_lines(staff_pixels, img)
        img = self.close_gaps(img)
        seg_line = (self.staff_lines[9] + self.staff_lines[10])/2
        img = img[0:seg_line,:]
        self.mark_components(img)
        cv2.imwrite(self.data_dir + "staff_removed.png",img)

if __name__=="__main__":
    data_dir = "/Users/ambermadvariya/data/omr_cv/"
    fname = "sample.png"
    pre_inst = preProcess(data_dir)
    pre_inst.run(fname)
    