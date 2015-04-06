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
from skimage import io
from skimage import color
from skimage.morphology import  remove_small_objects, convex_hull_image, binary_dilation, binary_erosion
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.measure import label
from scipy.ndimage.measurements import center_of_mass
from skimage.draw import circle
import subprocess as sp
from moviepy.config import get_setting

from moviepy.tools import subprocess_call

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
    '''
    def cart2pol(self,x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)
    '''    
    def cart2pol(self,x, y):
        """returns r, theta(degrees0-360)
        """
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.degrees(np.arctan2(y, x))
        if theta<0:
            theta=360+theta
        return r, theta
        
    def openVid(self,path):
        if(path != ''):
            self.cur_frame=0
            try:
                self.clip=VideoFileClip(path,audio=False)  
            except:
                self.clip=self.force_open(path)
            self.duration = self.clip.duration
            self.parent.setTicks(self.duration)
            self.length = round(self.clip.fps*self.duration) + 1
            self.frame=self.clip.get_frame(self.cur_frame)
            #self.pixmap = QtGui.QPixmap.fromImage(array2qimage(frame))
            self.signal_update_image.emit()

    def force_open(self,path):
        cmd= [get_setting("FFMPEG_BINARY"), "-i", path,"-y", "-an", "-r", str(30),"-vcodec","copy",
             'tmp.avi']
        subprocess_call(cmd,verbose=True)
        return VideoFileClip('tmp.avi',audio=False)  
             
    
    def changeAOI(self,aoi):
        self.aoi = aoi
        self.aoi[0]=round((float((99-aoi[0]))/99)*self.bgnd.shape[0])
        self.aoi[1]=round((float((99-aoi[1]))/99)*self.bgnd.shape[0])
        self.aoi[2]=round((float(aoi[2])/99)*self.bgnd.shape[1])
        self.aoi[3]=round((float(aoi[3])/99)*self.bgnd.shape[1])
            
    def open_bgrnd(self,path):
        image=color.rgb2gray(io.imread(path))
        self.set_bgrnd(image)

    def set_bgrnd(self,image):
        self.bgnd=image
        if self.aoi==[0,0,1,1]:
            self.aoi=[0,self.bgnd.shape[0],0,self.bgnd.shape[1]]
        
    def open_mask(self,path):
        self.mask=color.rgb2gray(io.imread(path))
        x=(np.array([self.mask[:,i].sum() for i in range(self.mask.shape[1])])!=0)
        self.aoi[2]=np.where(x==True)[0][0]
        self.aoi[3]=np.where(x==True)[0][-1]        
        y=(np.array([self.mask[i,:].sum() for i in range(self.mask.shape[0])])!=0)
        self.aoi[0]=np.where(y==True)[0][0]
        self.aoi[1]=np.where(y==True)[0][-1]
        
            

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
        
        self.cur_frame=(float(sliderPos)/999)*(self.length)
        
        self.frame=self.clip.get_frame(self.cur_frame/self.clip.fps)
        #self.pixmap = QtGui.QPixmap.fromImage(array2qimage(frame))
        self.signal_update_image.emit()
        
    def play(self):
        
        if( self.cur_frame < self.length):
            self.frame=self.clip.get_frame(self.cur_frame/self.clip.fps)
            #self.pixmap = QtGui.QPixmap.fromImage(array2qimage(frame))
            self.signal_update_image.emit()
            self.cur_frame+=1
            self.parent.frameSlider.setValue(round(float(999*self.cur_frame)/float(self.length)))
            self.parent.currentTimeLbl.setText(str(round((self.parent.frameSlider.value()/999.0)*self.duration,2)))
            
            #QtCore.QCoreApplication.processEvents()
            
    def process_all(self,t_start,t_end,step=None):
        
        processclip=self.clip.subclip(t_start=t_start, t_end=t_end)
        duration=processclip.duration
        self.currentLength=round(self.clip.fps*duration) + 1
        if self.processing:
            self.parent.pbar.show()
            skipped=0
            status=0
            result=[]
            time=[]
            if step==None:
                for frame in processclip.iter_frames():
                    if self.processing:
                        a,b=self.process_image(frame)
                        skipped+=a
                        time=float(status)*self.clip.fps
                        result.append([time,b]) 
                        status+=1
                        self.parent.pbar.setValue(float(status)*100/self.currentLength)
                    else:
                        return 
            else:
                for i in np.arange(0,processclip.duration,step):
                    if self.processing:
                        frame=processclip.get_frame(i)
                        a,b=self.process_image(frame)
                        skipped+=a
                        result.append([i,b])
                        status+=1
                        self.parent.pbar.setValue(float(status)*step*100/duration)
                    else:
                        return
                    
            self.processing=False
            print("Done! " + str(skipped)+ " frames skipped!" )
            self.parent.pbar.hide()
            self.process_data(result)
             
        else:
            self.parent.pbar.hide()
                #QtCore.QCoreApplication.processEvents()
    def process_data(self,result):
        i=0
        empty=[]
        for line in result:
            if line[1]==None:
                empty.append(i)
            i+=1    
        if len(empty)!=0:
            filtered=np.delete(result,empty,0)
        else:
            filtered=np.array(result,dtype=object)
        step=filtered[1,0]-filtered[0,0]
        times=np.asarray(filtered[:,0].tolist())
        coord=np.asarray(filtered[:,1].tolist())
        dataframe=np.column_stack((times,coord))
        x_unit=self.parent.coordx_box.value()
        y_unit=self.parent.coordy_box.value()
        dataframe[:,1]*=y_unit/float(self.aoi[1]-self.aoi[0])
        dataframe[:,2]*=x_unit/float(self.aoi[3]-self.aoi[2])
        time=dataframe[-1,0]
        distance=np.sum(np.linalg.norm(dataframe[1:,1:]-dataframe[:-1,1:],axis=1))
        speed=distance/float(time)
        

        cartez = np.array([[line[2]-float(x_unit/2),-(line[1]-float(y_unit/2))] for line in dataframe])
        polar = np.array([self.cart2pol(line[2]-float(x_unit/2),-(line[1]-float(y_unit/2))) for line in dataframe])
        home=polar[-1,1]
        polar_refer=polar.copy()
        polar_refer[:,1]-=(home-360/36)
        polar_refer[:,1]*=18/360.0
        polar_refer[:,1]=polar_refer[:,1].astype(int)
        i=0
        for line in polar_refer:
            if line[1]<-8:
                 polar_refer[i,1]=line[1]+18
            elif line[1]>9:
                polar_refer[i,1]=line[1]-18
            if line[0]<30:
                polar_refer[i,1]=100
            i+=1
        result_dict={'Total':{'Distance':distance,'Time':time,'Speed':speed}}
        subresult = [[x,polar_refer[:,1].tolist().count(x)] for x in set(polar_refer[:,1].tolist())]
        for i in subresult:
            if i[0]==100:
                result_dict['Center']={'Time':i[1]*step}
            else:
                result_dict[str(int(i[0]))]={'Time':i[1]*step}
        self.parent.table.buildFromDict(result_dict,['Distance','Time','Speed'],[])
            
        
        #print polar_refer
        
    def process_current(self):
        if self.process_image(self.clip.get_frame(self.cur_frame/self.clip.fps)) == 1:
            print("Object not found!" )
            
    def process_image(self,frame):

        substracted = (self.get_aoi(color.rgb2gray(frame),self.aoi) - self.get_aoi(self.bgnd,self.aoi))
        masked = substracted*(self.get_aoi(color.rgb2gray(self.mask),self.aoi)!=0)
        binarized= ( masked > self.trhd)
        
        clean=self.fill_holes(remove_small_objects(binarized,self.obj_area))
        
        result=self.erosion_N_times(clean,self.diamond,self.shrink)
        
        com=center_of_mass(result)
        
        QtCore.QCoreApplication.processEvents()
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
                pass
            
            if np.isnan(com[0]) or np.isnan(com[1]):
                return 1,None
            else:
                print com
                return 0,com
               
            
        
        
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
