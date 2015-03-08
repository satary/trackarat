# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:58:12 2015

@author: armeev
"""
from moviepy.editor import *
from qimage2ndarray import *
import numpy as np
import scipy.ndimage
from PyQt4 import QtGui, QtCore
from PyQt4.QtOpenGL import *
from skimage import io
from skimage import color
from skimage.morphology import  remove_small_objects, convex_hull_image, binary_dilation, binary_erosion
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.measure import label
from scipy.ndimage.measurements import center_of_mass
from skimage.draw import circle

class PreviewPlayer(QtCore.QObject): 
    signal_update_image = QtCore.pyqtSignal()  
    finished = QtCore.pyqtSignal()    
    def __init__(self,parent):
        QtCore.QObject.__init__(self)
        self.cur_frame=0
        self.playing = False
        self.videoLoaded = False
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.play)
        self.parent=parent
        self.processing = False
        self.disp_proc_pic = True
        self.trhd = 0.1
        self.obj_area = 200
        self.shrink =2
        #self.mainMask
        self.aoi=[0,0,1,1]
        self.output_style=4
        self.diamond = np.array([0,1,0,1,1,1,0,1,0], dtype=bool).reshape((3,3))
        
    def openVid(self,path):
        if(path != ''):
            self.cur_frame=0
            self.clip=VideoFileClip(path)  
            self.length = round(self.clip.fps*self.clip.duration) + 1
            self.frame=self.clip.get_frame(self.cur_frame)
            #self.pixmap = QtGui.QPixmap.fromImage(array2qimage(frame))
            self.signal_update_image.emit()
            
    def changeAOI(self,aoi):
        self.aoi = aoi
        self.aoi[0]=round((float((99-aoi[0]))/99)*self.bgnd.shape[0])
        self.aoi[1]=round((float((99-aoi[1]))/99)*self.bgnd.shape[0])
        self.aoi[2]=round((float(aoi[2])/99)*self.bgnd.shape[1])
        self.aoi[3]=round((float(aoi[3])/99)*self.bgnd.shape[1])
            
    def open_bgrnd(self,path):
        self.bgnd=color.rgb2gray(io.imread(path))
        self.aoi=[0,self.bgnd.shape[0],0,self.bgnd.shape[1]]
        
            

    def start(self):
        self.timer.start(1000.0/self.clip.fps)
        #QtCore.QCoreApplication.processEvents()
    
    def stop(self):
        self.timer.stop()
        self.cur_frame=0
        self.frame=self.clip.get_frame(self.cur_frame)
        #self.pixmap = QtGui.QPixmap.fromImage(array2qimage(frame))
        self.signal_update_image.emit()
        self.parent.frameSlider.setValue(0)
  
    def pause(self):
        self.timer.stop()

        
    def showframe(self,sliderPos):
        
        self.cur_frame=(float(sliderPos)/99)*(self.length)
        
        self.frame=self.clip.get_frame(self.cur_frame/self.clip.fps)
        #self.pixmap = QtGui.QPixmap.fromImage(array2qimage(frame))
        self.signal_update_image.emit()
        
    def play(self):
        
        if( self.cur_frame < self.length):
            self.frame=self.clip.get_frame(self.cur_frame/self.clip.fps)
            #self.pixmap = QtGui.QPixmap.fromImage(array2qimage(frame))
            self.signal_update_image.emit()
            self.cur_frame+=1
            self.parent.frameSlider.setValue(round(float(99*self.cur_frame)/float(self.length)))
            
            #QtCore.QCoreApplication.processEvents()
            
    def process_all(self):
        
        if self.processing:
            self.parent.pbar.show()
            self.skipped=0
            self.status=0
            for frame in self.clip.iter_frames():
                if self.processing:
                    self.process_image(frame)
                else:
                    return
                    
            self.processing=False
            print("Done! " + str( self.skipped)+ " frames skipped!" )
            self.parent.pbar.hide()
        else:
            self.parent.pbar.hide()
                #QtCore.QCoreApplication.processEvents()
            
    def process_current(self):
        self.skipped=0
        self.status=0
        self.process_image(self.clip.get_frame(self.cur_frame/self.clip.fps))
        if self.skipped==1:
            print("Object not found!" )
            
    def process_image(self,frame):

        substracted = (self.get_aoi(color.rgb2gray(frame),self.aoi) - self.get_aoi(self.bgnd,self.aoi))
        binarized= ( substracted > self.trhd)
        
        clean=self.fill_holes(remove_small_objects(binarized,self.obj_area))
        
        result=self.erosion_N_times(clean,self.diamond,self.shrink)
        
        com=center_of_mass(result)
        QtCore.QCoreApplication.processEvents()
        self.status+=1
        if self.disp_proc_pic:
            try:
                if self.output_style==0:    
                    self.frame = substracted*255
                if self.output_style==1:  
                    self.frame = binarized*255
                if self.output_style==2:  
                    self.frame = clean*255
                if self.output_style==3:  
                    self.frame = result*255
                elif self.output_style==4:
                
                    img = self.get_aoi(np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8),self.aoi)
                    rr, cc = circle(com[0], com[1], 5)
                    img[rr, cc] = 255
                    mask = img > 10
                    merged= self.get_aoi(color.rgb2gray(frame),self.aoi)*np.invert(mask)
                    
                    self.frame = merged*255
                
                self.signal_update_image.emit()
            except:
                self.skipped+=1
        self.parent.pbar.setValue(float(self.status)*100/self.length)
        
    def get_aoi(self,grayscale_img,aoi):
        
        return grayscale_img[aoi[0]:aoi[1],aoi[2]:aoi[3]]
    
    def fill_holes(self,binary):
        '''Fill holes in binary image'''
    
        labels = label(binary)
        labelcount = np.bincount(labels.ravel())
        bg = np.argmax(labelcount)
        binary[labels != bg] = True
        return binary
        
    def erosion_N_times(self,binary, selem, N):
        '''Returns the result of two sequential binary erosions'''

        for i in range(int(N)):
            binary = binary_erosion(binary, selem)
    
        return binary
        
    def close(self):
        self.processing=False
        self.timer.stop()
        self.finished.emit()