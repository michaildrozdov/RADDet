# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import cv2
import numpy as np
import tensorflow as tf
import time
import sys
import json

import shutil
from glob import glob
from tqdm import tqdm

import model.head_YOLO as yolohead
from PositionFromArucoVideo.RadarDataLoader import RadarDataLoader
from PositionFromArucoVideo.RawDataLoaderEx import RawDataLoaderEx
from PythonRadarTracker.tracker import Tracker
from PythonRadarTracker.radar_data import RadarData

import metrics.mAP as mAP

import util.loader as loader
import util.helper as helper
import util.drawer as drawer

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
from threading import Thread
import math

RETRY_PERIOD = 250 # ms

def load_network(instance, path):
    print(f"Loading network: {path}")
    instance.network = tf.saved_model.load(path)

def load_raw_data(instance, path, txnum, slide):
    print(f"Loading raw data: {path}")
    instance.radarDataLoader = RawDataLoaderEx(path, txnum, txnum==1, slide)

def load_intermediate(instance):
    print(f"Loading intermediate data: {instance.videoPath}")
    instance.radarDataLoader = RadarDataLoader(instance.videoPath,
                                            1.0 / instance.slide,
                                            instance.dopplerPadding,
                                            2,
                                            samplesToLoad=15000,
                                            randomize=False,
                                            showSynchronizedVideo=True,
                                            finishedFileCallback=instance.finishDataLoading)
    instance.radarDataLoader.logLevel = 3

    instance.totalRadarFrames = instance.radarDataLoader.get_total_samples()

# Original implementation taken from:
# https://stackoverflow.com/questions/63698714/how-to-show-markings-on-qdial-in-pyqt5-python
# and extended.
class ValueDial(QWidget):
    _dialProperties = ('minimum', 'maximum', 'value', 'singleStep', 'pageStep',
        'notchesVisible', 'tracking', 'wrapping', 
        'invertedAppearance', 'invertedControls', 'orientation')
    _inPadding = 3
    _outPadding = 2
    valueChanged = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        # remove properties used as keyword arguments for the dial
        dialArgs = {k:v for k, v in kwargs.items() if k in self._dialProperties}
        for k in dialArgs.keys():
            kwargs.pop(k)
        super().__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        self.dial = QDial(self, **dialArgs)
        layout.addWidget(self.dial)
        #self.dial.valueChanged.connect(self.valueChanged)
        self.dial.valueChanged.connect(self.modifyAndReemit)
        # make the dial the focus proxy (so that it captures focus *and* key events)
        self.setFocusProxy(self.dial)

        # simple "monkey patching" to access dial functions
        self.value = self.dial.value
        #self.setValue = self.dial.setValue
        self.minimum = self.dial.minimum
        self.maximum = self.dial.maximum
        self.wrapping = self.dial.wrapping
        self.notchesVisible = self.dial.notchesVisible
        self.setNotchesVisible = self.dial.setNotchesVisible
        self.setNotchTarget = self.dial.setNotchTarget
        self.notchSize = self.dial.notchSize
        self.invertedAppearance = self.dial.invertedAppearance
        self.setInvertedAppearance = self.dial.setInvertedAppearance
        self.showLogarithmic = False
        self.updateSize()

    def modifyAndReemit(self, value):
        if self.showLogarithmic and value:
            absoluteValue = abs(value)
            transformed = (int)(math.pow(10, absoluteValue / 10.0))
            if value > 0:
                self.valueChanged.emit(transformed)
            else:
                self.valueChanged.emit(-transformed)
        else:
            # Send without modifications
            self.valueChanged.emit(value)

    def inPadding(self):
        return self._inPadding

    def setInPadding(self, padding):
        self._inPadding = max(0, padding)
        self.updateSize()

    def outPadding(self):
        return self._outPadding

    def setOutPadding(self, padding):
        self._outPadding = max(0, padding)
        self.updateSize()

    # the following functions are required to correctly update the layout
    def setMinimum(self, minimum):
        self.dial.setMinimum(minimum)
        self.updateSize()

    def setMaximum(self, maximum):
        self.dial.setMaximum(maximum)
        self.updateSize()

    def setValue(self, value):
        if self.showLogarithmic and value:
            absoluteValue = abs(value)
            transformed = 10 * math.log10(value)
            self.dial.setValue(transformed)
        else:
            self.dial.setValue(value)

    def setWrapping(self, wrapping):
        self.dial.setWrapping(wrapping)
        self.updateSize()
    
    def setShowLogarithmic(self, state):
        self.showLogarithmic = state

    def updateSize(self):
        # a function that sets the margins to ensure that the value strings always
        # have enough space
        fm = self.fontMetrics()
        minWidth = max(fm.width(str(v)) for v in range(self.minimum(), self.maximum() + 1))
        self.offset = max(minWidth, fm.height()) / 2
        margin = int(self.offset + self._inPadding + self._outPadding)
        self.layout().setContentsMargins(margin, margin, margin, margin)

    def translateMouseEvent(self, event):
        # a helper function to translate mouse events to the dial
        return QtGui.QMouseEvent(event.type(), 
            self.dial.mapFrom(self, event.pos()), 
            event.button(), event.buttons(), event.modifiers())

    def changeEvent(self, event):
        if event.type() == QtCore.QEvent.FontChange:
            self.updateSize()

    def mousePressEvent(self, event):
        self.dial.mousePressEvent(self.translateMouseEvent(event))

    def mouseMoveEvent(self, event):
        self.dial.mouseMoveEvent(self.translateMouseEvent(event))

    def mouseReleaseEvent(self, event):
        self.dial.mouseReleaseEvent(self.translateMouseEvent(event))

    def paintEvent(self, event):
        radius = min(self.width(), self.height()) / 2
        radius -= (self.offset / 2 + self._outPadding)
        invert = -1 if self.invertedAppearance() else 1
        if self.wrapping():
            angleRange = 360
            startAngle = 270
            rangeOffset = 0
        else:
            angleRange = 300
            startAngle = 240 if invert > 0 else 300
            rangeOffset = 1
        fm = self.fontMetrics()

        # a reference line used for the target of the text rectangle
        reference = QtCore.QLineF.fromPolar(radius, 0).translated(self.rect().center())
        fullRange = self.maximum() - self.minimum()
        textRect = QtCore.QRect()

        qp = QtGui.QPainter(self)
        qp.setRenderHints(qp.Antialiasing)
        totalVisibleNotches = (int)(fullRange + rangeOffset) / self.notchSize()
        notchMultiplier = 1
        if totalVisibleNotches > 10:
            notchMultiplier = (int)((totalVisibleNotches + 10) / 10)
        for p in range(0, fullRange + rangeOffset, notchMultiplier * self.notchSize()):
            value = self.minimum() + p
            if invert < 0:
                value -= 1
                if value < self.minimum():
                    continue

            if self.showLogarithmic and value:
                absoluteValue = abs(value)
                transformed = (int)(math.pow(10, absoluteValue / 10.0))
                if value > 0:
                    value = transformed
                else:
                    value = -transformed

            angle = p / fullRange * angleRange * invert
            reference.setAngle(startAngle - angle)
            textRect.setSize(fm.size(QtCore.Qt.TextSingleLine, str(value)))
            textRect.moveCenter(reference.p2().toPoint())
            qp.drawText(textRect, QtCore.Qt.AlignCenter, str(value))


# A custom class for the image area. Maybe we can use Qt drawing primitives instead of OpenCV
class CustomImage(QWidget):
    def __init__(self, parent=None):
        super(CustomImage, self).__init__(parent)
        self.image = None
    
    def setImage(self, image):
        self.image = image
        self.setMinimumSize(image.size())
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0,0), self.image)
        qp.end()

class ApplicationWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Radar player")
        self.setGeometry(0, 0, 1920, 800)
        self.setMinimumSize(960, 500)

        self.scrollArea = QScrollArea()
        self.scrollArea.setMaximumSize(1900, 600)
        self.scrollArea.verticalScrollBar().setFixedWidth(10)
        self.scrollArea.horizontalScrollBar().setFixedHeight(10)
        self.scrollArea.setWidgetResizable(True)

        self.videoFrame = CustomImage(self)
        self.scrollArea.setWidget(self.videoFrame)
        self.setCentralWidget(self.scrollArea)
        self.video = cv2.VideoCapture()

        self.createMenu()
        self.createStatusBar()
        self.createControls()

        self.playing = False
        self.jumpby = 10 # In radar data slides
        self.usesBpm = True
        self.useTracking = False

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.updateWithIncrement)
        self.resultImage = np.zeros((1, 1, 1), np.uint8)

        self.readConfig()

        self.radarDataLoader = None
        self.slide = 32
        self.totalRadarFrames = 0
        self.totalVideoFrames = 0

        self.chirpPeriod = 11 # ms
        self.videoPeriod = 50 # ms, TODO: Get it from the video itself
        self.videoStartIndex = 0 # index
        self.radarStartIndex = 0 # index

        self.loadDefaultSynchronizationData()
        self.tracker = Tracker(0.5) # Average sensitivity
        self.distDisplace = 8 # Our free parameter due to cables (?) in indices
        self.usingIntermediate = False # Ignore the video reader or the synchronization data as
                                       # the data loader will provide matching frames.
        self.justFinishedData = False
        self.loadingLabel = QLabel(self.scrollArea)
        self.loadingMovie = QtGui.QMovie("./images/giphy.gif")
        self.loadingLabel.setGeometry(720, 180, 480, 400)
        self.loadingLabel.setMovie(self.loadingMovie)
        self.network = None

        self.loadingTimer = QtCore.QTimer(self)
        self.loadingTimer.timeout.connect(self.networkLoaded)
        self.dataLoadingTimer = QtCore.QTimer(self)
        self.dataLoadingTimer.timeout.connect(self.dataLoaded)
        self.intermediateLoadingTimer = QtCore.QTimer(self)
        self.intermediateLoadingTimer.timeout.connect(self.intermediateLoaded)
        self.loadDefaultNetwork()

    def videoIndexFromRadar(self, radarIndex):
        radarIndex = max(radarIndex, self.radarStartIndex)
        print(f"radarIndex {radarIndex}, self.radarStartIndex {self.radarStartIndex}, self.videoStartIndex {self.videoStartIndex}, video index {(radarIndex - self.radarStartIndex) * self.chirpPeriod // self.videoPeriod + self.videoStartIndex}")
        return (radarIndex - self.radarStartIndex) * self.chirpPeriod \
            // self.videoPeriod + self.videoStartIndex

    def updateSlider(self, value):
        if not self.video.isOpened():
            return
        if not self.radarDataLoader:
            return
        if self.indirectSliderUpdate: # Some other code updates the slider
            return

        total = self.radarDataLoader.get_total_samples()
        newIndex = total * value // value
        self.radarDataLoader.set_frame_index(newIndex)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.videoIndexFromRadar(newIndex*self.radarDataLoader.get_shift_per_sample()))

        #total = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        #self.video.set(cv2.CAP_PROP_POS_FRAMES, total / 100.0 * value)
        self.update()
        self.repaint()

    def loadRadarSynchronizationData(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "Select radar synchronization data file", "(*.json)", options=options)

        if filename:
            print(f"File {filename} was selected as a radar synchronization data file")
            with open(filename, 'r') as f:
                self.radarAnnotations = json.load(f)
                self.hasRadarAnnotations = True

    def loadVideoSynchronizationData(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "Select video synchronization data file", "(*.json)", options=options)

        if filename:
            print(f"File {filename} was selected as a video synchronization data file")
            with open(filename, 'r') as f:
                self.videoAnnotations = json.load(f)
                self.hasVideoAnnotations = True

    def loadDefaultSynchronizationData(self):
        with open("annotated.json", 'r') as f:
            self.videoAnnotations = json.load(f)
            self.hasVideoAnnotations = True
        with open("radar_annotated.json", 'r') as f:
            self.radarAnnotations = json.load(f)
            self.hasRadarAnnotations = True

    # Load something other than current configured network
    def loadNetwork(self):
        dir = QFileDialog.getExistingDirectory(self, "Select saved network directory")

        if dir and not self.networkLoadingThread.is_alive():
            self.loadingLabel.show()
            self.loadingMovie.start()
            self.disablePlaying()
            print(f"Network will be loaded from {dir}")
            self.networkLoadingThread = Thread(target=load_network, args=(self, dir,))
            self.loadingTimer.start(RETRY_PERIOD)
            self.networkLoadingThread.start()
            #self.network = tf.saved_model.load(dir)
        return
    
    def loadDefaultNetwork(self):
        self.loadingLabel.show()
        self.loadingMovie.start()
        self.disablePlaying()
        self.networkLoadingThread = Thread(target=load_network, args=(self, "model_b_" + str(self.configTrain["batch_size"]) + \
            "lr_" + str(self.configTrain["learningrate_init"]),))
        self.loadingTimer.start(RETRY_PERIOD)
        self.networkLoadingThread.start()
        #self.network = load_network(tf.saved_model.load("model_b_" + str(self.configTrain["batch_size"]) + \
        #           "lr_" + str(self.configTrain["learningrate_init"])))

    def networkLoaded(self):
        if not self.networkLoadingThread.is_alive():
            self.loadingLabel.hide()
            self.loadingMovie.stop()
            self.loadingTimer.stop()
            self.enablePlaying()

    def intermediateLoaded(self):
        if self.dataLoadingThread.is_alive():
            return

        self.loadingLabel.hide()
        self.loadingMovie.stop()
        self.intermediateLoadingTimer.stop()
        self.setFrameIndex(0)
        self.update()
        self.enablePlaying()

    def dataLoaded(self):
        if self.dataLoadingThread.is_alive():
            return

        self.loadingLabel.hide()
        self.loadingMovie.stop()
        self.dataLoadingTimer.stop()

        self.totalRadarFrames = self.radarDataLoader.get_total_samples()
        print(f"Opened radar file with {self.totalRadarFrames} frames")
        print(f"Final path of the video: {self.videoPath}")

        if self.video.isOpened():
            self.video.release()

        self.usingIntermediate = False
        self.video = cv2.VideoCapture(self.videoPath)
        self.totalVideoFrames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.updateSynchronization()

        self.progressSlider.setValue(0)
        self.setFrameIndex(0)
        videoFrameIndex = self.videoIndexFromRadar((self.frameIndex + 1)*self.radarDataLoader.get_shift_per_sample())
        print(f"Calculated video frame index was {videoFrameIndex} from {self.frameIndex}")
        self.video.set(cv2.CAP_PROP_POS_FRAMES, videoFrameIndex - 1)
        print(f"Opened video file with {self.totalVideoFrames} frames")
        self.update()
        self.enablePlaying()

    def openRadarRaw(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        path, _ = QFileDialog.getOpenFileName(self, "Select radar raw file", "","(*.raw)", options=options)
        # TODO: Allow to play without the matching video
        if path:
            self.loadingLabel.show()
            self.loadingMovie.start()
            self.disablePlaying()
            self.lastDataPath = path
            print(f"Selected raw radar data {path}")
            self.lastDirIndex = path.rfind('/')
            filename = path[self.lastDirIndex + 1:]
            self.currentRadarFilename = filename

            rawExtensionIndex = filename.rfind('.raw')
            if rawExtensionIndex < 0:
                return

            bpmIdentifierIndex = filename.rfind('_bpm')
            tx2IdentifierIndex = filename.rfind('_tx2')
            if bpmIdentifierIndex >= 0:
                # it is BPM signal
                self.isBpmBox.setChecked(True)
                filename = filename[:bpmIdentifierIndex]
                #self.radarDataLoader = RawDataLoaderEx(path, 3, False, self.slide)
                self.dataLoadingThread = Thread(target=load_raw_data, args=(self, path, 3, self.slide,))
                self.dataLoadingTimer.start(RETRY_PERIOD)
                self.dataLoadingThread.start()

            elif tx2IdentifierIndex >= 0:
                self.isBpmBox.setChecked(False)
                filename = filename[:tx2IdentifierIndex]
                #self.radarDataLoader = RawDataLoaderEx(path, 1, True, self.slide)
                self.dataLoadingThread = Thread(target=load_raw_data, args=(self, path, 1, self.slide,))
                self.dataLoadingTimer.start(RETRY_PERIOD)
                self.dataLoadingThread.start()
            else:
                print(f"Filename is of unknown naming convention: {filename}")
                return

            self.currentVideoFilename = "output_" + filename + "_wall.avi"

            print(f"Opened radar file with {self.totalRadarFrames} frames")

            if self.topViewRadio.isChecked():
                self.currentVideoFilename = "output_" + filename + "_fish.avi"
            else:
                self.currentVideoFilename = "output_" + filename + "_wall.avi"

            self.videoPath = path[:self.lastDirIndex] + "/" + self.currentVideoFilename

    def openRadarIntermediate(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        path, _ = QFileDialog.getOpenFileName(self, "Select radar intermediate data file", "","(*.npy)", options=options)

        if path:
            self.loadingLabel.show()
            self.loadingMovie.start()
            self.disablePlaying()
            print(f"Selected intermediate radar data {path}")
            lastDirIndex = path.rfind('/')
            filename = path[lastDirIndex + 1:]

            tx2IdentifierIndex = filename.rfind('_tx2')
            extensionIndex = filename.rfind('.npy')
            if tx2IdentifierIndex < 0 or extensionIndex < 0:
                print(f"Wrong file when loading the intermediate data '{path}'")
                return

            filename = filename[:tx2IdentifierIndex]
            print(f"Got filename as date '{filename}'")
            if filename + '_tx2.raw' in self.radarAnnotations:
                self.currentRadarFilename = filename + '_tx2.raw'
            elif filename + '_bpm.raw' in self.radarAnnotations:
                self.currentRadarFilename = filename + '_bpm.raw'

            # RadarDataLoader always loads the wall camera video
            self.currentVideoFilename = "output_" + filename + "_wall.avi"
            self.videoPath = path[:lastDirIndex] + "/" + self.currentVideoFilename
            print(f"Full video path {self.videoPath}")

            self.dataLoadingThread = Thread(target=load_intermediate, args=(self,))
            self.intermediateLoadingTimer.start(RETRY_PERIOD)
            self.dataLoadingThread.start()

            '''
            self.radarDataLoader = RadarDataLoader(self.videoPath,
                                                   1.0 / self.slide,
                                                   self.dopplerPadding,
                                                   2,
                                                   samplesToLoad=15000,
                                                   randomize=False,
                                                   showSynchronizedVideo=True,
                                                   finishedFileCallback=self.finishDataLoading)
            self.radarDataLoader.logLevel = 2

            self.totalRadarFrames = self.radarDataLoader.get_total_samples()
            '''
            self.usingIntermediate = True
            self.video = cv2.VideoCapture(self.videoPath)
            self.totalVideoFrames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Should not be needed
            #self.updateSynchronization()

            self.progressSlider.setValue(0)
            #self.setFrameIndex(0)
            #self.update()

    def finishDataLoading(self):
        print("Calling the finishDataLoading callback")
        self.justFinishedData = True

    def getFrameIndex(self):
        self.frameIndex = self.radarDataLoader.get_frame_index()
        '''
        temp = self.video.get(cv2.CAP_PROP_POS_FRAMES)
        if temp >= 0:
            self.frameIndex = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            print(f"cv2.CAP_PROP_POS_FRAME property does not return a valid index")
        '''
        return self.frameIndex

    def setFrameIndex(self, frameIndex):
        self.frameIndex = frameIndex
        self.radarDataLoader.set_frame_index(frameIndex)

        # TODO: Check about -1, what if we calculate 0?
        videoFrameIndex = self.videoIndexFromRadar((self.frameIndex + 1)*self.radarDataLoader.get_shift_per_sample())
        if videoFrameIndex - 1 < self.video.get(cv2.CAP_PROP_FRAME_COUNT):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, videoFrameIndex - 1)
        else:
            self.frameIndex = 0
            self.radarDataLoader.set_frame_index(frameIndex)
            videoFrameIndex = self.videoIndexFromRadar((self.frameIndex + 1)*self.radarDataLoader.get_shift_per_sample())
            self.video.set(cv2.CAP_PROP_POS_FRAMES, videoFrameIndex - 1)
        #self.video.set(cv2.CAP_PROP_POS_FRAMES, self.frameIndex - 1)
        
        self.update()

    def stopAtPrev(self):
        toSet = max(0.0, self.getFrameIndex() - 2)
        self.setFrameIndex(toSet)

    def stopAtNext(self):
        toSet = min(self.totalRadarFrames - 1, self.getFrameIndex())
        self.setFrameIndex(toSet)

    def jumpPrev(self):
        toSet = max(0.0, self.getFrameIndex() - self.jumpby)
        self.setFrameIndex(toSet)

    def jumpNext(self):
        toSet = min(self.totalRadarFrames - 1, self.getFrameIndex() + self.jumpby)
        self.setFrameIndex(toSet)

    def updateWithIncrement(self):
        print(f"In updateWithIncrement self.frameIndex {self.frameIndex}")
        self.setFrameIndex(self.getFrameIndex()) # No need for +1 since the loader handles this
        print(f"In updateWithIncrement self.frameIndex(2) {self.frameIndex}")

    def update(self):
        if not self.radarDataLoader:
            return
        if not self.video or not self.video.isOpened() and not self.usingIntermediate:
            return

        gtCenters = []
        if self.usingIntermediate:
            RAD, gt, frame = next(self.radarDataLoader.yield_next_data())
            isBpm = self.radarDataLoader.is_bpm() # This loader only provides this flag after
                                                  # at least one frame is created by it.
            self.isBpmBox.setChecked(isBpm)
            for box in gt["boxes"]:
                gtCenters.append((box[0], box[1], box[2]))
            ret = True
        else:
            ret, frame = self.video.read()

        if not ret:
            if self.frameIndex > self.totalRadarFrames and self.totalRadarFrames:
                self.frameIndex = self.totalRadarFrames - 1
                self.video.set(cv2.CAP_PROP_POS_FRAMES, self.videoIndexFromRadar((self.frameIndex + 1)*self.radarDataLoader.get_shift_per_sample()) - 1)
                ret, frame = self.video.read()
                self.runOrStop()
                if not ret:
                    return # Still failing

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dim = (int(frame.shape[1] / 4), int(frame.shape[0] / 4))

        self.resultImage = np.zeros((dim[1], 2 * dim[0], 3), np.uint8)    

        self.resultImage[:, dim[0]:, :] = cv2.resize(frame, dim)
        draw_radar_axes(self.resultImage,
                        (self.resultImage.shape[0], self.resultImage.shape[1] / 2),
                        (270 - self.maxAzimuth, 270 + self.maxAzimuth))
        
        videoFrame = np.array([])
        
        if not self.usingIntermediate:
            RAD, RED, dummyGt = self.radarDataLoader.produce_frame()

        RAD = np.log(np.abs(RAD) + 1e-9)
        RAD = (RAD - self.configData["global_mean_log"]) / self.configData["global_variance_log"]
        data = tf.expand_dims(tf.constant(RAD, dtype=tf.float32), axis=0)

        if data != None and self.network != None:
            feature = modelAsGraph(self.network, data)

            pred_raw, pred = yolohead.boxDecoder(feature, self.inputSize, \
                self.anchorBoxes, self.numClasses, self.yoloheadXyzScales[0])
  
            pred_frame = pred[0]
            predicitons = helper.yoloheadToPredictions(pred_frame, \
                                    conf_threshold=self.configEvaluate["confidence_threshold"])

            nms_pred = helper.nms(predicitons, \
                                    self.configInference["nms_iou3d_threshold"], \
                                    self.configModel["input_shape"], \
                                    sigma=0.3, method="nms")

            if self.useTracking:
                # Do tracking
                dataFrame = []
                for detection in nms_pred:
                    confidence = detection[6]
                    r = (detection[0]-self.distDisplace) * self.dr
                    velocity = (detection[2] - self.dopplerPadding / 2) * self.dvel
                    if self.usesBpm:
                        velocity /= 3.0
                    azimuth = np.arcsin(
                        (detection[1] - self.azPadding / 2) * self.dsinAz) * 180 / np.pi
                    elevation = 0.0

                    #print(f"Adding r={r:0.2f}, velocity={velocity:0.2f}, azimuth={azimuth:0.2f}")
                    dataFrame.append(RadarData(r, velocity, np.radians(azimuth), np.radians(elevation), confidence))
                dt = self.chirpPeriod * self.radarDataLoader.get_shift_per_sample() / 1000.0
                self.tracker.update(dataFrame, dt)

                # Draw tracks
                activeTracks = self.tracker.get_active_tracks()
                draw_tracks(self.resultImage,
                            (self.resultImage.shape[0], self.resultImage.shape[1] / 2),
                            activeTracks)

            else:
                draw_detections(self.resultImage,
                                (self.resultImage.shape[0], self.resultImage.shape[1] / 2),
                                nms_pred,
                                self.azPadding,
                                self.dr,
                                self.dsinAz,
                                self.distDisplace)
        if len(gtCenters):
            draw_gt(self.resultImage,
                    (self.resultImage.shape[0], self.resultImage.shape[1] / 2),
                    gtCenters,
                    self.azPadding,
                    self.dr,
                    self.dsinAz,
                    self.distDisplace)

        image = QtGui.QImage(self.resultImage,
                             self.resultImage.shape[1],
                             self.resultImage.shape[0],
                             self.resultImage.strides[0],
                             QtGui.QImage.Format_RGB888)

        self.videoWidth = frame.shape[1] # TODO: use me
        self.videoHeight = frame.shape[0] # TODO: use me

        self.videoFrame.setImage(image)

        #sliderPosition = int(self.video.get(cv2.CAP_PROP_POS_FRAMES) - 1 * 100.0 / self.totalFrames)
        if self.justFinishedData:
            sliderPosition = 0
            self.justFinishedData = False
        else:
            sliderPosition = self.radarDataLoader.get_frame_index() * 100 // self.totalRadarFrames
        if sliderPosition >= 0 and sliderPosition <= 100:
            self.indirectSliderUpdate = True
            self.progressSlider.setValue(sliderPosition)
            self.indirectSliderUpdate = False
        print(f"self.frameIndex at the end {self.frameIndex}")

    def runOrStop(self):
        if self.playing:
            self.playing = False
            self.timer.stop()
            self.playStopButton.setText('Play')
        else:
            self.playing = True
            self.timer.start(50)
            self.playStopButton.setText('Stop')

    def createControls(self):
        self.controls = QWidget(self)
        self.controls.setGeometry(0, 610, 1900, 180)

        controlsLayout = QGridLayout()
        self.jumpBackButton = QPushButton('|<<', parent=self.controls)
        self.jumpForwardButton = QPushButton('>>|', parent=self.controls)
        self.playStopButton = QPushButton('Play', parent=self.controls)
        self.forwardButton = QPushButton('>|', parent=self.controls)
        self.backButton = QPushButton('|<', parent=self.controls)

        self.progressSlider = QSlider(QtCore.Qt.Horizontal, parent=self.controls)
        self.progressSlider.sliderMoved[int].connect(self.updateSlider)

        #self.trackingSensitivitySlider = QSlider(QtCore.Qt.Horizontal, self.controls)
        self.trackingSensitivitySlider = ValueDial(minimum=0, maximum=100)
        self.trackingSensitivitySlider.setNotchesVisible(True)
        self.trackingSensitivitySlider.setValue(90)
        self.trackingSensitivitySlider.valueChanged.connect(self.updateTrackingSensitivity)
        self.trackingSensitivitySlider.setEnabled(False)

        # TODO: The user min/max directly as he wants (-10000/10000 in this case) regardless
        # if he intends to use setShowLogarithmic() or not.
        #self.radarDelaySlider = ValueDial(minimum=-40, maximum=40)
        #self.radarDelaySlider.setShowLogarithmic(True)
        self.radarDelaySlider = ValueDial(minimum=-5000, maximum=5000)

        self.radarDelaySlider.setNotchesVisible(True)
        self.radarDelaySlider.setValue(0)
        self.radarDelaySlider.valueChanged.connect(self.updateSynchronizationUser)
        self.radarDelaySlider.setEnabled(False)

        self.isBpmBox = QCheckBox('Is BPM', parent=self.controls)
        self.isBpmBox.setChecked(True)
        self.isBpmBox.setEnabled(False)
        self.isBpmBox.stateChanged.connect(self.updateIsBpm)

        self.useTrackingBox = QCheckBox('Use tracking', parent=self.controls)
        self.useTrackingBox.setChecked(False)
        self.useTrackingBox.stateChanged.connect(self.updateUseTracking)

        self.topViewRadio = QRadioButton('Top view', parent=self.controls)
        self.topViewRadio.setChecked(False)
        self.sideViewRadio = QRadioButton('Side view', parent=self.controls)
        self.sideViewRadio.setChecked(True)
        self.viewGroup = QButtonGroup(self.controls)
        self.viewGroup.addButton(self.topViewRadio)
        self.viewGroup.addButton(self.sideViewRadio)

        self.trackingSensitivityLabel = QLabel(self.controls)
        self.trackingSensitivityLabel.setText('Tracking sensitivity 90%')

        self.radarDelayLabel = QLabel(self.controls)
        self.radarDelayLabel.setText('Radar start delay (0 ms)')

        controlsLayout.addWidget(self.progressSlider, 0, 0, 1, 13)
        controlsLayout.addWidget(self.jumpBackButton, 1, 0)
        controlsLayout.addWidget(self.backButton, 1, 1)
        controlsLayout.addWidget(self.playStopButton, 1, 2)
        controlsLayout.addWidget(self.forwardButton, 1, 3)
        controlsLayout.addWidget(self.jumpForwardButton, 1, 4)
        controlsLayout.addWidget(self.trackingSensitivityLabel, 1, 6)
        #controlsLayout.addWidget(self.trackingSensitivitySlider, 1, 7, 1, 3)
        controlsLayout.addWidget(self.trackingSensitivitySlider, 1, 7)
        controlsLayout.addWidget(self.radarDelayLabel, 1, 8)
        controlsLayout.addWidget(self.radarDelaySlider, 1, 9)
        controlsLayout.addWidget(self.topViewRadio, 1, 11)
        controlsLayout.addWidget(self.sideViewRadio, 2, 11)
        controlsLayout.addWidget(self.isBpmBox, 1, 12)
        controlsLayout.addWidget(self.useTrackingBox, 2, 12)

        self.controls.setLayout(controlsLayout)

        self.playStopButton.clicked.connect(self.runOrStop)
        self.backButton.clicked.connect(self.stopAtPrev)
        self.forwardButton.clicked.connect(self.stopAtNext)
        self.jumpBackButton.clicked.connect(self.jumpPrev)
        self.jumpForwardButton.clicked.connect(self.jumpNext)

    def updateIsBpm(self, value):
        self.usesBpm = value

    def updateUseTracking(self, value):
        self.useTracking = value

        if value:
            print(f"Creating a tracker with sensitivity {self.trackingSensitivitySlider.value() / 100.0}")
            self.tracker = Tracker(self.trackingSensitivitySlider.value() / 100.0)
            self.trackingSensitivitySlider.setEnabled(True)
        else:
            self.trackingSensitivitySlider.setEnabled(False)

    def disablePlaying(self):
        self.jumpBackButton.setDisabled(True)
        self.jumpForwardButton.setDisabled(True)
        self.playStopButton.setDisabled(True)
        self.forwardButton.setDisabled(True)
        self.backButton.setDisabled(True)

    def enablePlaying(self):
        self.jumpBackButton.setDisabled(False)
        self.jumpForwardButton.setDisabled(False)
        self.playStopButton.setDisabled(False)
        self.forwardButton.setDisabled(False)
        self.backButton.setDisabled(False)

    def createMenu(self):
        self.menu = self.menuBar().addMenu('&Menu')
        self.menu.addAction('&Load radar data', self.openRadarRaw)
        self.menu.addAction('&Load from intermediate', self.openRadarIntermediate)
        self.menu.addAction('&Load radar synchronization', self.loadRadarSynchronizationData)
        self.menu.addAction('&Load video synchronization', self.loadVideoSynchronizationData)
        self.menu.addAction('&Load network', self.loadNetwork)
        self.menu.addAction('&Exit', self.close)

        self.frameSlidingMenu = self.menuBar().addMenu('&Frame sliding')
        self.frameSlidingMenu.addAction('&Set sliding default', self.slidingDefault)
        self.frameSlidingMenu.addAction('&Set sliding half default', self.slidingHalfDefault)
        self.frameSlidingMenu.addAction('&Set sliding 1', self.sliding1)
        self.frameSlidingMenu.addAction('&Set sliding 2', self.sliding2)
        self.frameSlidingMenu.addAction('&Set sliding 3', self.sliding3)
        self.frameSlidingMenu.addAction('&Set sliding 4', self.sliding4)
        self.frameSlidingMenu.addAction('&Set sliding 5', self.sliding5)

    def createStatusBar(self):
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('Some important messages are shown here')

    def readConfig(self):
        config = loader.readConfig()
        self.configData = config["DATA"]
        self.configRadar = config["RADAR_CONFIGURATION"]
        self.configModel = config["MODEL"]
        self.configTrain = config["TRAIN"]
        self.configEvaluate = config["EVALUATE"]
        self.configInference = config["INFERENCE"]

        self.maxAzimuth = self.configRadar["azimuth_size"] * self.configRadar["angular_resolution"] * 90.0 / np.pi
        
        self.inputSize = list(self.configModel["input_shape"]) # TODO: get this from the model input instead
        self.anchorBoxes = loader.readAnchorBoxes() # load anchor boxes with order
        self.numClasses = len(self.configData["all_classes"])
        self.yoloheadXyzScales = self.configModel["yolohead_xyz_scales"]

        self.dsinAz = c / (freq0 * self.configRadar["azimuth_size"] * d)
        self.dr = self.configRadar["range_resolution"]
        self.dvel = self.configRadar["velocity_resolution"]

        self.azPadding = self.configRadar["azimuth_size"]
        self.dopplerPadding = self.configRadar["doppler_size"]

    def setSliding(self, value):
        self.slide = value

        if self.radarDataLoader:
            self.radarDataLoader.set_shift_per_sample(value)
            self.totalRadarFrames = self.radarDataLoader.get_total_samples()

        self.setFrameIndex(0)

    def slidingDefault(self):
        self.setSliding(self.inputSize[2])

    def slidingHalfDefault(self):
        self.setSliding(self.inputSize[2] // 2)

    def sliding1(self):
        self.setSliding(1)

    def sliding2(self):
        self.setSliding(2)

    def sliding3(self):
        self.setSliding(3)

    def sliding4(self):
        self.setSliding(4)

    def sliding5(self):
        self.setSliding(5)

    def updateSynchronization(self):
        radarStartMs = -1
        cameraStartMs = -1
        if self.hasRadarAnnotations and self.hasVideoAnnotations:
            if self.currentRadarFilename in self.radarAnnotations:
                startRadar = self.radarAnnotations[self.currentRadarFilename]["start"]
                radarAnnotationPeriod = self.chirpPeriod
                if self.usesBpm:
                    radarAnnotationPeriod = 3 * self.chirpPeriod
                radarStartMs = startRadar * radarAnnotationPeriod
                print(f"Radar start for current file {startRadar} or {radarStartMs} ms")
            else:
                print(f"Don't have radar annotations for {self.currentRadarFilename}")

            if self.currentVideoFilename in self.videoAnnotations:
                startVideo = self.videoAnnotations[self.currentVideoFilename]["start"][0]
                cameraStartMs = startVideo * self.videoPeriod
                print(f"Video start for current file {startVideo} or {cameraStartMs} ms")
            else:
                print(f"Don't have video annotations for {self.currentVideoFilename}")
        if radarStartMs == -1 or cameraStartMs == -1:
            print("Couldn't find one of the annotations. Can't synchronize")
            self.radarDelaySlider.setEnabled(True)
        else:
            print(f"radarStartMs {radarStartMs} ms, cameraStartMs {cameraStartMs} ms")
            self.radarDelaySlider.setEnabled(False)
            if cameraStartMs >= radarStartMs:
                self.radarStartIndex = 0
                self.videoStartIndex = (cameraStartMs - radarStartMs) // self.videoPeriod
            else:
                self.videoStartIndex = 0
                self.radarStartIndex = (radarStartMs - cameraStartMs) // (self.chirpPeriod * self.radarDataLoader.get_shift_per_sample())

            print(f"self.radarStartIndex {self.radarStartIndex}, self.videoStartIndex {self.videoStartIndex}")

    # By how much radar trails the video. If negative, all actions on the radar signal happen sooner.
    def updateSynchronizationUser(self, radarDelayMs: int):
        print(f"Got delay value from dial {radarDelayMs}")

        self.radarDelayLabel.setText(f'Radar start delay ({radarDelayMs} ms)')
        cameraStartMs = 0
        radarStartMs = radarDelayMs
        if radarStartMs < 0:
            cameraStartMs = -radarStartMs
            radarStartMs = 0

        if cameraStartMs >= radarStartMs:
            self.radarStartIndex = 0
            self.videoStartIndex = (cameraStartMs - radarStartMs) // self.videoPeriod
            print("cameraStartMs >= radarStartMs")
        else:
            self.videoStartIndex = 0
            print("radarStartMs >= cameraStartMs")
            if self.radarDataLoader != None:
                self.radarStartIndex = (radarStartMs - cameraStartMs) // (self.chirpPeriod * self.radarDataLoader.get_shift_per_sample())
        print(f"self.radarStartIndex {self.radarStartIndex}, self.videoStartIndex {self.videoStartIndex}")

    def updateTrackingSensitivity(self, value):
        # TODO: Just pass the new sensitivity to rearrange thresholds without clearing the tracking state
        print(f"Creating a tracker with sensitivity {self.trackingSensitivitySlider.value() / 100.0}")
        self.tracker = Tracker(self.trackingSensitivitySlider.value() / 100.0)

        self.trackingSensitivityLabel.setText(f'Tracking sensitivity {self.trackingSensitivitySlider.value()}%')


c = 0.3 # Speed of light
d = 0.061 # Array step
freq0 = 3.1 # Frequency start

def rad_to_cartesian(rangeIndex, angleIndex, anglePadding, dr, dSinAz):
    r = rangeIndex * dr
    a = np.arcsin((angleIndex - anglePadding/2) * dSinAz)

    x = r * np.sin(a)
    y = r * np.cos(a)
    return (x, y)

def rad_to_cartesian_simple(r, a):
    x = r * np.sin(a)
    y = r * np.cos(a)
    return (x, y)

def cutImage(image_dir, image_filename):
    image_name = os.path.join(image_dir, image_filename)
    image = cv2.imread(image_name)
    part_1 = image[:, 1540:1750, :]
    part_2 = image[:, 2970:3550, :]
    part_3 = image[:, 4370:5400, :]
    part_4 = image[:, 6200:6850, :]
    new_img = np.concatenate([part_1, part_2, part_3, part_4], axis=1)
    cv2.imwrite(image_name, new_img)


def cutImage3Axes(image_dir, image_filename):
    image_name = os.path.join(image_dir, image_filename)
    image = cv2.imread(image_name)
    part_1 = image[:, 1780:2000, :]
    part_2 = image[:, 3800:4350, :]
    part_3 = image[:, 5950:6620, :]
    new_img = np.concatenate([part_1, part_2, part_3], axis=1)
    cv2.imwrite(image_name, new_img)

def cutImageCustom(image_dir, image_filename):
    image_name = os.path.join(image_dir, image_filename)
    image = cv2.imread(image_name)
    #part_1 = image[:, 860:1023, :]
    #part_2 = image[:, 1928:2133, :]
    #part_3 = image[:, 2726:3570, :]
    part_1 = image[:, 850:1043, :]
    part_2 = image[:, 1918:2143, :]
    part_3 = image[:, 2726:3570, :]
    new_img = np.concatenate([part_1, part_2, part_3], axis=1)
    cv2.imwrite(image_name, new_img)


def loadDataForPlot(dataLoader, config_data, showSynchronizedVideo, \
                    config_radar, interpolation=15.):
    """ Load data one by one for generating evaluation images """
    sequence_num = -1
    totalSamples = dataLoader.get_total_samples()
    #for RAD_file in all_RAD_files:
    for s in range(totalSamples):
        sequence_num += 1

        if showSynchronizedVideo:
            RAD_complex, gt_instances, stereo_left_image = next(dataLoader.yield_next_data())
        else:
            RAD_complex, gt_instances = next(dataLoader.yield_next_data())

        RA = helper.getSumDim(np.abs(RAD_complex), target_axis=-1)
        RD = helper.getSumDim(np.abs(RAD_complex), target_axis=1)
        
        RA_cart = helper.toCartesianMaskCustom(RA, config_radar)
        RA_img = helper.norm2Image(RA)[..., :3]
        RD_img = helper.norm2Image(RD)[..., :3]
        RA_cart_img = helper.norm2Image(RA_cart)[..., :3]

        if not showSynchronizedVideo:
            img_file = "images/calib_target.jpg" # Just a dummy
            stereo_left_image = loader.readStereoLeft(img_file)

        RAD_data = np.log(np.abs(RAD_complex) + 1e-9)
        RAD_data = (RAD_data - config_data["global_mean_log"]) / \
                            config_data["global_variance_log"]
        data = tf.expand_dims(tf.constant(RAD_data, dtype=tf.float32), axis=0)
        yield sequence_num, data, stereo_left_image, RD_img, RA_img, RA_cart_img, gt_instances

def draw_radar_axes(image, draw_area, angles):

    max_distance = (image.shape[0] - 50)
    # Draw range circles
    for ind in range(1, 6):
        image = draw_half_circle_rounded(image,
                                    draw_area,
                                    angles, 
                                    ind * max_distance / 5)

    # Draw lines
    height, width = draw_area                                
    center = (int(width / 2), int(height - 25))
    angle_max = np.pi * (angles[1] - 270) / 180.0
    dx = int(np.sin(angle_max) * max_distance)
    dy = int(np.cos(angle_max) * max_distance)

    image = cv2.line(image, center, (center[0] + dx, center[1] - dy), (255, 255, 255), 2)
    image = cv2.line(image, center, (center[0] - dx, center[1] - dy), (255, 255, 255), 2)

    # Center
    image = cv2.line(image, center, (center[0], center[1] - max_distance), (255, 255, 255), 1)

    step = np.pi * 20 / 180.0
    for angle in np.arange(step, angle_max, step):
        dx = int(np.sin(angle) * max_distance)
        dy = int(np.cos(angle) * max_distance)

        image = cv2.line(image, center, (center[0] + dx, center[1] - dy), (255, 255, 255), 1)
        image = cv2.line(image, center, (center[0] - dx, center[1] - dy), (255, 255, 255), 1)

    return image

def draw_half_circle_rounded(image, draw_area, angles, radius):
    height, width = draw_area
    # Ellipse parameters
    center = (int(width / 2), int(height - 25))
    axes = (int(radius), int(radius))
    angle = 0
    startAngle, endAngle = angles
    thickness = 2

    return cv2.ellipse(img=image,
                center=center,
                axes=axes,
                angle=angle,
                startAngle=startAngle,
                endAngle=endAngle,
                color=(255, 255, 255),
                thickness=thickness)

def draw_detections(image, draw_area, detections, anglePadding, dr, dsin, distDisplace=0):
    #print(f"Drawing:\n{detections}")
    maxRange = draw_area[0] - 50
    zeroAt = draw_area[1] / 2
    for det in detections:
        (x, y) = rad_to_cartesian(det[0] - distDisplace, det[1], anglePadding, dr, dsin)
        center = (int(zeroAt - (maxRange * x)/5), int(draw_area[0] - 25 - y * maxRange / 5))
        image = cv2.circle(image,
                           center,
                           5,
                           color=(255,60,60),
                           thickness = 2)
        drawLoc = (center[0] + 10, center[1] - 10)
        image = cv2.putText(image, f'{det[6]:0.2f}', drawLoc, cv2.FONT_HERSHEY_SIMPLEX,  
                   fontScale=0.5, color=(255,60,60), thickness=1, lineType=cv2.LINE_AA) 
    return image

def draw_gt(image, draw_area, gtCenters, anglePadding, dr, dsin, distDisplace=0):
    maxRange = draw_area[0] - 50
    zeroAt = draw_area[1] / 2
    for gt in gtCenters:
        (x, y) = rad_to_cartesian(gt[0] - distDisplace, gt[1], anglePadding, dr, dsin)
        center = (int(zeroAt - (maxRange * x)/5), int(draw_area[0] - 25 - y * maxRange / 5))
        image = cv2.circle(image,
                           center,
                           5,
                           color=(60,60,255),
                           thickness = 2)
        drawLoc = (center[0] + 10, center[1] - 10)
    return image

def draw_tracks(image, draw_area, tracks):
    maxRange = draw_area[0] - 50
    zeroAt = draw_area[1] / 2
    for track in tracks:
        state = track.state
        (x, y) = rad_to_cartesian_simple(state[0], state[2]) # In the track state azimuth has index 2
        center = (int(zeroAt - (maxRange * x)/5), int(draw_area[0] - 25 - y * maxRange / 5))
        image = cv2.circle(image,
                           center,
                           5,
                           color=(255,60,60),
                           thickness = 2)
        drawLoc = (center[0] + 10, center[1] - 10)
        image = cv2.putText(image, f'{track.get_average_confidence():0.2f}', drawLoc, cv2.FONT_HERSHEY_SIMPLEX,  
                   fontScale=0.5, color=(255,60,60), thickness=1, lineType=cv2.LINE_AA) 
    return image

@tf.function
def modelAsGraph(model, data):
    return model(data)

def main():
    ### NOTE: GPU manipulation, you may can print this out if necessary ###
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    config = loader.readConfig()
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]
    config_evaluate = config["EVALUATE"]
    config_inference = config["INFERENCE"]

    anchor_boxes = loader.readAnchorBoxes() # load anchor boxes with order
    anchor_cart = loader.readAnchorBoxes(anchor_boxes_file="./anchors_cartboxes.txt")
    num_classes = len(config_data["all_classes"])
    yolohead_xyz_scales = config_model["yolohead_xyz_scales"]

    max_azimuth = config_radar["azimuth_size"] * config_radar["angular_resolution"] * 90.0 / np.pi

    dsinAz = c / (freq0 * config_radar["azimuth_size"] * d)

    ### NOTE: using the yolo head shape out from model for data generator ###
    '''
    model = M.RADDet(config_model, config_data, config_train, anchor_boxes)
    model.build([None] + config_model["input_shape"])
    model.backbone_stage.summary()
    model.summary()
    '''

    imported = tf.saved_model.load("model_b_" + str(config_train["batch_size"]) + \
                    "lr_" + str(config_train["learningrate_init"]))

    f = imported.signatures["serving_default"]

    cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)

    input_size = list(config_model["input_shape"]) # TODO: get this from the model input instead

    def inferencePlotting(all_original_files):
        """ Plot the predictions of all data in dataset """

        globalEndsList = [] # Last frames for each of the measurement
        globalIndex = 0
        def insert_file_end():
            globalEndsList.append(globalIndex)

        fig, axes = drawer.prepareFigure(3, figsize=(40, 6))
        colors = loader.randomColors(config_data["all_classes"])

        image_save_dir = "./images/inference_plots/"
        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir)
        else:
            shutil.rmtree(image_save_dir)
            os.makedirs(image_save_dir)
        print("Start plotting, it might take a while...")
        
        testLoader = RadarDataLoader(all_original_files,
                                    config_inference["loader_samples_ratio"],
                                    config_model["input_shape"][2],
                                    2,
                                    samplesToLoad=15000,
                                    randomize=False,
                                    showSynchronizedVideo=True,
                                    finishedFileCallback=insert_file_end)

        testLoader.logLevel = 3
        totalSamples = testLoader.get_total_samples()
        if not totalSamples:
            print('No data to process. Wrong "original_videos_pattern" value in config.json?')
            return

        pbar = tqdm(total=totalSamples)
        resultImage = np.zeros((1, 1, 1), np.uint8)

        for sequence_num, data, stereo_left_image, RD_img, RA_img, RA_cart_img, gt_instances in \
                loadDataForPlot(testLoader, config_data, True, config_radar):

            time1 = time.time()
            dim = (int(stereo_left_image.shape[1] / 4), int(stereo_left_image.shape[0] / 4))

            if resultImage.shape[0] < dim[1]:
                resultImage = np.zeros((dim[1], 2 * dim[0], 3), np.uint8)
            
            if data is None or stereo_left_image is None:
                pbar.update(1)
                continue

            RA_cart_img = stereo_left_image[:,:,::-1]
            
            time2 = time.time()

            feature = modelAsGraph(imported, data)

            time3 = time.time()

            pred_raw, pred = yolohead.boxDecoder(feature, input_size, \
                anchor_boxes, num_classes, yolohead_xyz_scales[0])
  
            time4 = time.time()
            pred_frame = pred[0]
            predicitons = helper.yoloheadToPredictions(pred_frame, \
                                    conf_threshold=config_evaluate["confidence_threshold"])

            time5 = time.time()
            nms_pred = helper.nms(predicitons, \
                                    config_inference["nms_iou3d_threshold"], \
                                    config_model["input_shape"], \
                                    sigma=0.3, method="nms")

            time6 = time.time()
            nms_pred_cart = None
            #drawer.clearAxes(axes)
            #drawer.drawInference(RD_img, \
            #        RA_img, RA_cart_img, nms_pred, \
            #        config_data["all_classes"], colors, axes, \
            #        radar_cart_nms=nms_pred_cart)
            time7 = time.time()
            #drawer.drawGt(RD_img, RA_img, gt_instances, axes)
            time8 = time.time()
            #drawer.saveFigure(image_save_dir, "%.6d.png"%(sequence_num))
            time9 = time.time()
            #img = drawer.getImage(fig)
            #cutImageCustom(image_save_dir, "%.6d.png"%(sequence_num))
            
            resultImage[:, dim[0]:, :] = cv2.resize(stereo_left_image, dim)

            draw_radar_axes(resultImage,
                            (resultImage.shape[0], resultImage.shape[1] / 2),
                            (270 - max_azimuth, 270 + max_azimuth))

            draw_detections(resultImage, (resultImage.shape[0], resultImage.shape[1] / 2), nms_pred, dsinAz)

            cv2.imshow('Display', resultImage)
            time10 = time.time()
            cv2.waitKey(1)

            print(f"Time dif 2-1 = {time2 - time1}")
            print(f"Time dif 3-2 = {time3 - time2}")
            print(f"Time dif 4-3 = {time4 - time3}")
            print(f"Time dif 5-4 = {time5 - time4}")
            print(f"Time dif 6-5 = {time6 - time5}")
            print(f"Time dif 7-6 = {time7 - time6}")
            print(f"Time dif 8-7 = {time8 - time7}")
            print(f"Time dif 9-8 = {time9 - time8}")
            print(f"Time dif 10-9 = {time10 - time9}")

    ### NOTE: inference starting from here ###
    all_original_files = "../new_raw_data/*/*_wall.avi"
    inferencePlotting(all_original_files)


if __name__ == "__main__":
    #main()
    app = QApplication(sys.argv)
    window = ApplicationWindow()
    window.show()
    sys.exit(app.exec_())
