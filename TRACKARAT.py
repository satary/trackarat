# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:58:12 2015

@author: armeev
"""
from previewplayer import *
from moviepy.editor import *
from qimage2ndarray import *
import numpy as np
import scipy.ndimage
from PyQt4 import Qt,QtGui, QtCore
from skimage import io
from skimage import color
from skimage.morphology import  remove_small_objects, convex_hull_image, binary_dilation, binary_erosion
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.measure import label
from scipy.ndimage.measurements import center_of_mass
from skimage.draw import circle





class Window(QtGui.QMainWindow):
    def __init__(self):

        #c = cv2.VideoCapture(0)
        QtGui.QWidget.__init__(self)

        self.setWindowTitle('Track\'a\'Rat')
        self.playThread = QtCore.QThread()
        self.player = PreviewPlayer(self)
        self.player.moveToThread(self.playThread)

        self.player.finished.connect(self.playThread.quit)
        self.player.signal_update_image.connect(self.update_image)

        #self.playThread.started.connect(self.player.startCapture)
        self.playThread.start()

### SETTINGS AREA
        self.openvideo_button = QtGui.QPushButton('Open File',self)
        self.openvideo_button.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DialogOpenButton))
        self.openvideo_button.clicked.connect(self.open_video_file)

        self.openbgrnd_button = QtGui.QPushButton('Open Background',self)
        self.openbgrnd_button.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DialogOpenButton))
        self.openbgrnd_button.clicked.connect(self.open_bgrnd)
        
        self.openmask_button = QtGui.QPushButton('Open Mask',self)
        self.openmask_button.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DialogOpenButton))
        self.openmask_button.clicked.connect(self.open_mask)

        self.start_processing_button = QtGui.QPushButton('Start / Cancel',self)
        self.start_processing_button.clicked.connect(self.start_processing)
        
        self.process_current_button = QtGui.QPushButton('Calculate current',self)
        self.process_current_button.clicked.connect(self.process_current)

        self.pbar = QtGui.QProgressBar(self)

        treshhold_label = QtGui.QLabel(self)
        treshhold_label.setText("Threshhold:")
        self.treshhold_box = QtGui.QDoubleSpinBox()
        self.treshhold_box.setRange(0,1)
        self.treshhold_box.setSingleStep(0.05)
        self.treshhold_box.setDecimals(2)
        self.treshhold_box.setValue(0.1)
        self.treshhold_box.valueChanged.connect(self.setSettings)


        obj_area_label = QtGui.QLabel(self)
        obj_area_label.setText("Object area:")
        self.obj_area_box = QtGui.QDoubleSpinBox()
        self.obj_area_box.setRange(0,4000)
        self.obj_area_box.setDecimals(0)
        self.obj_area_box.setValue(200)
        self.obj_area_box.valueChanged.connect(self.setSettings)

        shrink_label = QtGui.QLabel(self)
        shrink_label.setText("Remove small detail:")
        self.shrink_box = QtGui.QDoubleSpinBox()
        self.shrink_box.setRange(0,10)
        self.shrink_box.setDecimals(0)
        self.shrink_box.setValue(2)
        self.shrink_box.valueChanged.connect(self.setSettings)

        self.preview_checkbox = QtGui.QCheckBox('Preview result', self)
        self.preview_checkbox.setCheckState(2)
        self.preview_checkbox.stateChanged.connect(self.setSettings)

        self.preview_step_combo_box = QtGui.QComboBox(self)
        calculation_steps=['Substract background','Threshhold','Filtering small obj','Remooving small details','Result']
        self.preview_step_combo_box.addItems(calculation_steps)
        self.preview_step_combo_box.setCurrentIndex(4)
        self.preview_step_combo_box.currentIndexChanged.connect(self.setSettings)
        
        self.framebyframe_checkbox = QtGui.QCheckBox('Process every frame', self)
        self.framebyframe_checkbox.setCheckState(2)
        self.framebyframe_checkbox.stateChanged.connect(self.toogleProcessBox)
        
        self.process_label = QtGui.QLabel(self)
        self.process_label.setText("Process only (seconds)")
        self.process_label.hide()
        self.process_box = QtGui.QDoubleSpinBox()
        self.process_box.setRange(0,10)
        self.process_box.setDecimals(1)
        self.process_box.setValue(1)
        self.process_box.hide()


### PREVIEW
        self.scene = QtGui.QGraphicsScene()
        self.view = QtGui.QGraphicsView(self.scene)

        self.pixmap_item = QtGui.QGraphicsPixmapItem(QtGui.QPixmap(), None, self.scene)
        self.pixmap_item.mousePressEvent = self.pixelSelect
        self.view.setVerticalScrollBarPolicy(1)
        self.view.setHorizontalScrollBarPolicy(1)
        self.view.centerOn(1.0, 1.0)


        iconSize = QtCore.QSize(20, 20)

        self.playButton = QtGui.QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.setIconSize(iconSize)
        self.playButton.setToolTip("Play")
        self.playButton.clicked.connect(self.start)

        self.pauseButton = QtGui.QToolButton()
        self.pauseButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPause))
        self.pauseButton.setIconSize(iconSize)
        self.pauseButton.setToolTip("Pause")
        self.pauseButton.clicked.connect(self.player.pause)

        self.stopButton = QtGui.QToolButton()
        self.stopButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaStop))
        self.stopButton.setIconSize(iconSize)
        self.stopButton.setToolTip("Stop")
        self.stopButton.clicked.connect(self.player.stop)

        self.frameSlider = QtGui.QSlider(self)
        self.frameSlider.setMinimum(0)
        self.frameSlider.setMaximum(1000)
        self.frameSlider.setOrientation(QtCore.Qt.Horizontal)
        self.frameSlider.setTickPosition(QtGui.QSlider.TicksAbove)
        self.frameSlider.setTickInterval(200)
        self.frameSlider.sliderMoved.connect(self.goToFrame)
        
        videoticks = QtGui.QHBoxLayout()
        self.videoTicks = [QtGui.QLabel(self) for i in range(6)] 
               
        #self.videoTicks[0].setSizePolicy(QtGui.QSizePolicy.Fixed,QtGui.QSizePolicy.Minimum)
        self.videoTicks[-1].setAlignment(QtCore.Qt.AlignRight)
        #self.videoTicks[-1].setSizePolicy(QtGui.QSizePolicy.Fixed,QtGui.QSizePolicy.Minimum)
        [tick.setAlignment(QtCore.Qt.AlignCenter) for tick in self.videoTicks[1:-1]]

        
        [videoticks.addWidget(tick) for tick in self.videoTicks]
        videoticks.setStretchFactor(self.videoTicks[0],1)
        videoticks.setStretchFactor(self.videoTicks[-1],1)
        [videoticks.setStretchFactor(tick,2) for tick in self.videoTicks[1:-1]]
        
        
        startLbl=QtGui.QLabel('Start:')
        self.start_time_box = QtGui.QDoubleSpinBox()
        self.start_time_box.setRange(0,1)
        self.start_time_box.setSingleStep(1)
        self.start_time_box.setDecimals(1)
        self.start_time_box.setValue(0)
        startbtn=QtGui.QToolButton()
        startbtn.setText('<')
        startbtn.clicked.connect(self.setStartTime)
        
        EndLbl=QtGui.QLabel('End:')
        self.end_time_box = QtGui.QDoubleSpinBox()
        self.end_time_box.setRange(0,1)
        self.end_time_box.setSingleStep(1)
        self.end_time_box.setDecimals(1)
        self.end_time_box.setValue(0)
        endbtn=QtGui.QToolButton()
        endbtn.setText('<')
        endbtn.clicked.connect(self.setEndTime)
        
        self.currentTimeLbl=QtGui.QLabel(self)
        
        






### background and AOI

        self.l_crop_Slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.l_crop_Slider.valueChanged.connect(self.changeAOI)
        self.r_crop_Slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.r_crop_Slider.setValue(100)
        self.r_crop_Slider.valueChanged.connect(self.changeAOI)
        self.t_crop_Slider = QtGui.QSlider(QtCore.Qt.Vertical)
        self.t_crop_Slider.valueChanged.connect(self.changeAOI)
        self.b_crop_Slider = QtGui.QSlider(QtCore.Qt.Vertical)
        self.t_crop_Slider.setValue(100)
        self.b_crop_Slider.valueChanged.connect(self.changeAOI)
        self.bgng_label = QtGui.QLabel(self)
        self.bgng_label.setSizePolicy(QtGui.QSizePolicy.Expanding,QtGui.QSizePolicy.Expanding)





### compositing
        vbox1 = QtGui.QVBoxLayout()
        vbox1.addWidget(self.openvideo_button)
        vbox1.addWidget(self.openbgrnd_button)
        vbox1.addWidget(self.openmask_button)
        vbox1.addWidget(self.start_processing_button)
        vbox1.addWidget(self.process_current_button)




        vbox1.addWidget(treshhold_label)
        vbox1.addWidget(self.treshhold_box)
        vbox1.addWidget(obj_area_label)
        vbox1.addWidget(self.obj_area_box)
        vbox1.addWidget(shrink_label)
        vbox1.addWidget(self.shrink_box)
        vbox1.addWidget(self.preview_checkbox)
        vbox1.addWidget(self.preview_step_combo_box)
        vbox1.addWidget(self.framebyframe_checkbox)
        vbox1.addWidget(self.process_label)
        vbox1.addWidget(self.process_box)
        vbox1.addWidget(self.pbar)
        vbox1.setAlignment(QtCore.Qt.AlignTop)
        self.pbar.hide()

        SettingsArea=QtGui.QWidget()
        SettingsArea.setLayout(vbox1)

        tabs = QtGui.QTabWidget(self)
        tab1 = QtGui.QWidget()
        self.tab2 = QtGui.QWidget()
        tab3 = QtGui.QWidget()

        tabs.addTab(tab1,"Preview")
        tab1.resizeEvent = self.update_image
        tab1.resizeEvent = self.update_bgnd

        preview_hlot = QtGui.QHBoxLayout()
        preview_hlot.addWidget(self.playButton)
        preview_hlot.addWidget(self.pauseButton)
        preview_hlot.addWidget(self.stopButton)
        
        preview_hlot.addWidget(startLbl)
        preview_hlot.addWidget(self.start_time_box)
        preview_hlot.addWidget(startbtn)
        
        preview_hlot.addWidget(EndLbl)
        preview_hlot.addWidget(self.end_time_box)
        preview_hlot.addWidget(endbtn)
        preview_hlot.addSpacerItem(QtGui.QSpacerItem(20,0
            ,QtGui.QSizePolicy.Expanding,QtGui.QSizePolicy.Minimum))
        preview_hlot.addWidget(self.currentTimeLbl)
        preview_hlot.setAlignment(QtCore.Qt.AlignLeft)

        preview_vlot = QtGui.QVBoxLayout()
        preview_vlot.addWidget(self.view)
        #preview_vlot.addSpacerItem(QtGui.QSpacerItem(20,40
        #    ,QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Minimum))
        preview_vlot.addLayout(videoticks)
        preview_vlot.addWidget(self.frameSlider)
        preview_vlot.addLayout(preview_hlot)
        preview_vlot.setAlignment(QtCore.Qt.AlignTop)

        tab1.setLayout(preview_vlot)

        tabs.addTab(self.tab2,"Background and AoI")
        self.tab2.resizeEvent = self.update_bgnd

        bgnd_glot = QtGui.QGridLayout()

        bgnd_glot.addWidget(self.t_crop_Slider,1,0)
        bgnd_glot.addWidget(self.b_crop_Slider,1,2)
        bgnd_glot.addWidget(self.l_crop_Slider,0,1)
        bgnd_glot.addWidget(self.r_crop_Slider,2,1)

        bgnd_glot.addWidget(self.bgng_label,1,1)

        bgnd_vlot = QtGui.QVBoxLayout()
        bgnd_vlot.addLayout(bgnd_glot)

        bgnd_vlot.addSpacerItem(QtGui.QSpacerItem(20,40
            ,QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Minimum))


        self.tab2.setLayout(bgnd_vlot)


        tabs.addTab(tab3,"Masks")

        splitter1 = QtGui.QSplitter(QtCore.Qt.Horizontal)
        splitter1.addWidget(SettingsArea)
        splitter1.addWidget(tabs)
        splitter1.setSizes([120,480])
        splitter1.splitterMoved.connect(self.update_image)


        self.setCentralWidget(splitter1)
        self.setGeometry(100,100,640,480)
        self.show()
        
    def setStartTime(self):
        self.start_time_box.setValue(round((self.frameSlider.value()/999.0)*self.player.duration,2))
    
    def setEndTime(self):
        self.end_time_box.setValue(round((self.frameSlider.value()/999.0)*self.player.duration,2))
        
    def setTicks(self,length):        
        step=length/(len(self.videoTicks)-1)
        i=0
        for tick in self.videoTicks:
            tick.setText(str(round(step*i,1)))
            i+=1
            
        self.start_time_box.setRange(0,length)
        self.start_time_box.setSingleStep(1)
        self.start_time_box.setDecimals(1)
        self.start_time_box.setValue(0)
        
        self.end_time_box.setRange(0,length)
        self.end_time_box.setSingleStep(1)
        self.end_time_box.setDecimals(1)
        self.end_time_box.setValue(length)
    
    def setSettings(self):
        self.player.trhd = self.treshhold_box.value()
        self.player.obj_area = self.obj_area_box.value()
        self.player.shrink = self.shrink_box.value()
        self.player.disp_proc_pic = self.preview_checkbox.isChecked()
        self.player.output_style = self.preview_step_combo_box.currentIndex()
        if self.preview_checkbox.isChecked():
            self.preview_step_combo_box.show()
        else:
            self.preview_step_combo_box.hide()
            
        if not(self.player.processing):
            self.player.process_current()
            
    def toogleProcessBox(self):
        if self.framebyframe_checkbox.isChecked():
            self.process_box.hide()
            self.process_label.hide()
        else:
            self.process_box.show()
            self.process_label.show()

    def start(self):
        QtCore.QCoreApplication.processEvents()
        self.player.start()

    def pixelSelect(self, event):
        print event.pos()
        pen = QtGui.QPen(QtCore.Qt.red)
        self.scene.addEllipse(event.pos().x(), event.pos().y(), 2, 2, pen)

    def goToFrame(self):
        QtCore.QCoreApplication.processEvents()
        self.player.showframe(self.frameSlider.value()) 
        time=str(round((self.frameSlider.value()/999.0)*self.player.duration,2))
        self.currentTimeLbl.setText(time)
          

    def open_video_file(self):
        filename=unicode(QtGui.QFileDialog.getOpenFileName(self,
        'Open video file'))
        self.player.openVid(filename)

    def open_bgrnd(self):
        filename=unicode(QtGui.QFileDialog.getOpenFileName(self,
        'Open background file'))
        if(filename != ''):
            self.player.open_bgrnd(filename)
            self.update_bgnd(None)
    
    def open_mask(self):
        filename=unicode(QtGui.QFileDialog.getOpenFileName(self,
        'Open background file'))
        if(filename != ''):
            self.player.open_mask(filename)
            #self.update_bgnd(None)


    def start_processing(self):
        self.player.processing = not(self.player.processing)
        self.player.process_all(self.start_time_box.value(),self.end_time_box.value())
        
    def process_current(self):
        self.player.process_current()

    def update_image(self,event=None):
        
        try:
            pixmap=QtGui.QPixmap.fromImage(array2qimage(self.player.frame))
            #print self.player.frame.shape
            pix=pixmap.scaled(self.view.size(), QtCore.Qt.KeepAspectRatio)
            ofset = pix.size()-self.view.size()/2
            self.pixmap_item.setPixmap(pix)
            #self.pixmap_item.setOffset(5,5)
            #self.outbox.setPixmap(pixmap.scaled(self.outbox.size(), QtCore.Qt.KeepAspectRatio))
            self.pixmap_item.update()
            QtCore.QCoreApplication.processEvents()
        except:
            return

    def changeAOI(self):
        try:
            self.player.changeAOI([self.t_crop_Slider.value(),
                                   self.b_crop_Slider.value(),
                                   self.l_crop_Slider.value(),
                                   self.r_crop_Slider.value()])
            self.update_bgnd(None)
        except:
            return

    def update_bgnd(self,event):
        try:

            self.bgng_label.resize(self.tab2.size()-QtCore.QSize(60,110))

            mask=np.zeros((self.player.bgnd.shape[0],self.player.bgnd.shape[1]))

            mask[self.player.aoi[0]:self.player.aoi[1],self.player.aoi[2]:self.player.aoi[3]]=1

            pix=QtGui.QPixmap.fromImage(array2qimage(self.player.bgnd * mask*255))
            pix=pix.scaled(self.bgng_label.size(), QtCore.Qt.KeepAspectRatio)
            self.bgng_label.update()

            self.bgng_label.setPixmap(pix)
        except:
            return


    def closeEvent(self, event):
        self.player.close()
        self.playThread.quit()
        print "Closing TRACARAT, goodbye"




if __name__ == '__main__':

    import sys
    app = QtGui.QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
