import cv2
import numpy as np 
import os
from matplotlib import pyplot as plt
from scipy import ndimage
import Queue
from operator import itemgetter
import math

class preProcess:
    def __init__(self, data_dir, template_list):
        self.data_dir = data_dir
        self.template_list = template_list

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

    def mark_components(self, img, draw=True):
        self.components = self.find_connected_components(img)
        if draw:
            draw_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            for component in self.components:
                top, left, bottom, right = component
                cv2.rectangle(draw_img,(left,top),(right,bottom),(0,255,0),3)
            cv2.imwrite(self.data_dir + "components.png",draw_img)

    def mark_lines(self, img):
        draw_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        for component in self.components:
            top, left, bottom, right = component
            area = (bottom-top)*(right-left)
            if area <= 1: continue
            comp_img = img[top:bottom,left:right]
            for j in xrange(right-left):
                nnz = np.count_nonzero(comp_img[:,j])
                if float(nnz)/(bottom-top) < 0.4:
                    zero_indices = np.where(comp_img[:,j]==0)[0]
                    start_index = top + zero_indices[0]
                    last_index = top + zero_indices[zero_indices.shape[0]-1]
                    draw_img[start_index:last_index,left+j,:] = (0,0,255)
        return draw_img

    def match_template(self, img):
        draw_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        dedup_mat = np.zeros(img.shape,dtype=np.uint8)
        matches = []
        for template_name in self.template_list:
            template = cv2.imread(template_name,0)
            height, width = template.shape
            ratio = float(self.diff)/height
            template = cv2.resize(template, None, fx=ratio, fy=ratio, interpolation = cv2.INTER_CUBIC)
            template = self.binarize_file(template)
            template = 255 - template
            w, h = template.shape[::-1]
            res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
            window_size = int(math.floor(float(self.diff)/4))
            threshold = 0.55
            loc = np.where( res >= threshold)
            points = zip(*loc[::-1])
            points = sorted(points,key=lambda x:itemgetter(1,0))
            for pt in points:
                col, row = pt
                if (row-window_size)>=0 and (col-window_size)>=0 and (row+window_size)<img.shape[0] and (col+window_size)<img.shape[1]:
                    if np.sum(dedup_mat[row-window_size:row+window_size,col-window_size:col+window_size]) != 0:
                        dedup_mat[row,col] = 1
                        continue
                dedup_mat[row,col] = 1
                counter += 1
                matches.append((row+h/2,col+w/2))
                cv2.rectangle(draw_img, pt, (col + w, row + h), (0,0,255), 1)
        print len(matches)
        cv2.imwrite(self.data_dir + "templates.png",draw_img)
        matches = sorted(matches,key=itemgetter(1,0))
        return matches

    def remove_vertical_lines(self, img):
        height, width = img.shape
        for j in xrange(width):
            num_zero = height - np.count_nonzero(img[:,j])
            if num_zero > height/2: 
                img[:,j] = 255

    def parse_components(self, img, components, staff_lines):
        treble_letter_notes = []
        bass_letter_notes = []
        for comp in components:
            top, left, bottom, right = comp
            matches = self.match_template(img[top:bottom,left:right])
            if len(matches) == 0: continue
            matches = map(lambda x:(x[0]+top,x[1]+left),matches)

    def run(self, fname):
        img = self.open_file(fname)
        img = self.binarize_file(img)
        staff_pixels = self.detect_staff_pixels(img)
        self.remove_staff_lines(staff_pixels, img)
        img = self.close_gaps(img)
        cv2.imwrite(self.data_dir + "staff_removed.png",img)
        prev_seg_line = 0
        for i in xrange(len(self.staff_lines)/10):
            print i
            if 10*(i+1) >= len(self.staff_lines):
                seg_line = img.shape[0] - 1
            else:
                seg_line = (self.staff_lines[10*i+9] + self.staff_lines[10*(i+1)])/2
            ref_img = img[prev_seg_line:seg_line,:]
            self.remove_vertical_lines(ref_img)
            self.components = self.find_connected_components(ref_img)
            self.components = sorted(self.components,key=itemgetter(1,0))
            print self.components
            #self.mark_components(img)
            #self.match_template(img)    
            #prev_seg_line = seg_line

if __name__=="__main__":
    data_dir = "/Users/ambermadvariya/data/omr_cv/"
    fname = "sample.png"
    templates_dir = data_dir + "templates/"
    template_list = map(lambda x: templates_dir + x + ".png",["whole_note","quarter_note","half_note"])
    pre_inst = preProcess(data_dir, template_list)
    pre_inst.run(fname)
    