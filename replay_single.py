# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
import time
import sys

import shutil
from glob import glob
from tqdm import tqdm

import model.head_YOLO as yolohead
from PositionFromArucoVideo.RadarDataLoader import RadarDataLoader
from PositionFromArucoVideo.RawDataLoaderEx import RawDataLoaderEx
import metrics.mAP as mAP

import util.loader as loader
import util.helper as helper
import util.drawer as drawer

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *


# Case 1: Don't have synchronization data. Have to assume that video starts with the radar data
# Case 2: Have synchronization data. Keep two variables: radar start, video start.
#   One of them will be 0 and the other will express the difference between the latest started type
#   (e.g. camera recording started later, so camera) and current type (e.g. radar) in periods of the
#   current type. For example: camera started 300 ms later than the radar; video start = 0; radar start
#   is 300 // 11 = 27
#   later in its periods. 
# All sliders should work based on the radar data samples

chirpPeriod = 11 # ms
videoPeriod = 50 # ms, TODO: Get it from the video itself
videoStartIndex = 0 # index
radarStartIndex = 0 # index

def videoIndexFromRadar(radarIndex):
    radarIndex = max(radarIndex, radarStartIndex)
    return (radarIndex - radarStartIndex) * chirpPeriod // videoPeriod + videoStartIndex

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
        self.jumpby = 100 # In radar samples
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
        self.video.set(cv2.CAP_PROP_POS_FRAMES, videoIndexFromRadar(newIndex*self.radarDataLoader.get_shift_per_sample()))

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

    def loadVideoSynchronizationData(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "Select video synchronization data file", "(*.json)", options=options)

        if filename:
            print(f"File {filename} was selected as a video synchronization data file")

    # Load something other than current configured network
    def loadNetwork(self):
        dir = QFileDialog.getExistingDirectory(self, "Select saved network directory")

        if dir:
            print(f"Network will be loaded from {dir}")
            self.network = tf.saved_model.load(dir)
        return

    def openRadarRaw(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        path, _ = QFileDialog.getOpenFileName(self, "Select radar raw file", "","(*.raw)", options=options)

        videoFound = False
        # TODO: Allow to play without the matching video
        if path:
            print(f"Selected raw radar data {path}")
            lastDirIndex = path.rfind('/')
            filename = path[lastDirIndex + 1:]

            rawExtensionIndex = filename.rfind('.raw')
            if rawExtensionIndex < 0:
                return

            bpmIdentifierIndex = filename.rfind('_bpm')
            tx2IdentifierIndex = filename.rfind('_tx2')
            if bpmIdentifierIndex >= 0:
                # it is BPM signal
                self.isBpmBox.setChecked(True)
                filename = filename[:bpmIdentifierIndex]
                self.radarDataLoader = RawDataLoaderEx(path, 3, False, self.slide)
            elif tx2IdentifierIndex >= 0:
                self.isBpmBox.setChecked(False)
                filename = filename[:tx2IdentifierIndex]
                self.radarDataLoader = RawDataLoaderEx(path, 1, True, self.slide)
            else:
                print(f"Filename is of unknown naming convention: {filename}")
                return

            self.totalRadarFrames = self.radarDataLoader.get_total_samples()
            print(f"Opened radar file with {self.totalRadarFrames} frames")

            if self.topViewRadio.isChecked():
                filename = "output_" + filename + "_fish.avi"
            else:
                filename = "output_" + filename + "_wall.avi"

            self.videoPath = path[:lastDirIndex] + "/" + filename

            print(f"Final path of the video: {self.videoPath}")

            if self.video.isOpened():
                self.video.release()
            videoFound = True

        if videoFound:
            self.video = cv2.VideoCapture(self.videoPath)
            self.totalVideoFrames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.progressSlider.setValue(0)
            print(f"Opened video file with {self.totalVideoFrames} frames")
            self.update()

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
        videoFrameIndex = videoIndexFromRadar(self.frameIndex*self.radarDataLoader.get_shift_per_sample())
        print(f"Frame to set to the video {videoFrameIndex}")
        self.video.set(cv2.CAP_PROP_POS_FRAMES, videoFrameIndex - 1)
        #self.video.set(cv2.CAP_PROP_POS_FRAMES, self.frameIndex - 1)
        
        self.update()

    def stopAtPrev(self):
        toSet = max(0.0, self.getFrameIndex() - 1)
        self.setFrameIndex(toSet)

    def stopAtNext(self):
        toSet = min(self.totalRadarFrames - 1, self.getFrameIndex() + 1)
        self.setFrameIndex(toSet)

    def jumpPrev(self):
        toSet = max(0.0, self.getFrameIndex() - self.jumpby)
        self.setFrameIndex(toSet)

    def jumpNext(self):
        toSet = min(self.totalRadarFrames - 1, self.getFrameIndex() + self.jumpby)
        self.setFrameIndex(toSet)

    def updateWithIncrement(self):
        self.setFrameIndex(self.getFrameIndex() + 1)

    def update(self):
        if not self.radarDataLoader:
            return
        if not self.video or not self.video.isOpened():
            return

        print(f"Current video frame is {self.video.get(cv2.CAP_PROP_POS_FRAMES)}")

        ret, frame = self.video.read()
        if not ret:
            if self.frameIndex > self.totalRadarFrames and self.totalRadarFrames:
                self.frameIndex = self.totalRadarFrames - 1
                self.video.set(cv2.CAP_PROP_POS_FRAMES, videoIndexFromRadar(self.frameIndex*self.radarDataLoader.get_shift_per_sample()) - 1)
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
        
        RAD, RED, dummyGt = self.radarDataLoader.produce_frame()

        RAD = np.log(np.abs(RAD) + 1e-9)
        RAD = (RAD - self.configData["global_mean_log"]) / self.configData["global_variance_log"]
        data = tf.expand_dims(tf.constant(RAD, dtype=tf.float32), axis=0)

        if data != None and self.network != None:
            print("Have some data!")
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

            draw_detections(self.resultImage,
                            (self.resultImage.shape[0], self.resultImage.shape[1] / 2),
                            nms_pred,
                            self.configRadar["azimuth_size"],
                            self.dr,
                            self.dsinAz)

        image = QtGui.QImage(self.resultImage,
                             self.resultImage.shape[1],
                             self.resultImage.shape[0],
                             self.resultImage.strides[0],
                             QtGui.QImage.Format_RGB888)

        self.videoWidth = frame.shape[1] # TODO: use me
        self.videoHeight = frame.shape[0] # TODO: use me

        self.videoFrame.setImage(image)

        #sliderPosition = int(self.video.get(cv2.CAP_PROP_POS_FRAMES) - 1 * 100.0 / self.totalFrames)
        sliderPosition = self.radarDataLoader.get_frame_index() * 100 // self.totalRadarFrames
        if sliderPosition >= 0 and sliderPosition <= 100:
            self.indirectSliderUpdate = True
            self.progressSlider.setValue(sliderPosition)
            self.indirectSliderUpdate = False

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

        self.trackingSensitivitySlider = QSlider(QtCore.Qt.Horizontal, self.controls)
        self.trackingSensitivitySlider.setValue(90)

        self.isBpmBox = QCheckBox('Is BPM', parent=self.controls)
        self.isBpmBox.setChecked(True)
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
        self.trackingSensitivityLabel.setText('Tracking sensitivity')

        controlsLayout.addWidget(self.progressSlider, 0, 0, 1, 13)
        controlsLayout.addWidget(self.jumpBackButton, 1, 0)
        controlsLayout.addWidget(self.backButton, 1, 1)
        controlsLayout.addWidget(self.playStopButton, 1, 2)
        controlsLayout.addWidget(self.forwardButton, 1, 3)
        controlsLayout.addWidget(self.jumpForwardButton, 1, 4)
        controlsLayout.addWidget(self.trackingSensitivityLabel, 1, 6)
        controlsLayout.addWidget(self.trackingSensitivitySlider, 1, 7, 1, 3)
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

    def createMenu(self):
        self.menu = self.menuBar().addMenu('&Menu')
        self.menu.addAction('&Load radar data', self.openRadarRaw)
        self.menu.addAction('&Load radar synchronization', self.loadRadarSynchronizationData)
        self.menu.addAction('&Load video synchronization', self.loadVideoSynchronizationData)
        self.menu.addAction('&Load network', self.loadNetwork)
        self.menu.addAction('&Exit', self.close)

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
        self.network = tf.saved_model.load("model_b_" + str(self.configTrain["batch_size"]) + \
                   "lr_" + str(self.configTrain["learningrate_init"]))

        self.inputSize = list(self.configModel["input_shape"]) # TODO: get this from the model input instead
        self.anchorBoxes = loader.readAnchorBoxes() # load anchor boxes with order
        self.numClasses = len(self.configData["all_classes"])
        self.yoloheadXyzScales = self.configModel["yolohead_xyz_scales"]

        self.dsinAz = c / (freq0 * self.configRadar["azimuth_size"] * d)
        self.dr = self.configRadar["range_resolution"]

c = 0.3 # Speed of light
d = 0.061 # Array step
freq0 = 3.1 # Frequency start

def rad_to_cartesian(rangeIndex, angleIndex, anglePadding, dr, dSinAz):
    r = rangeIndex * dr
    a = np.arcsin((angleIndex - anglePadding/2) * dSinAz)

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

def draw_detections(image, draw_area, detections, anglePadding, dr, dsin):
    print(f"Drawing:\n{detections}")
    maxRange = draw_area[0] - 50
    zeroAt = draw_area[1] / 2
    for det in detections:
        (x, y) = rad_to_cartesian(det[0], det[1], anglePadding, dr, dsin)
        center = (int(zeroAt + (maxRange * x)/5), int(draw_area[0] - 25 - y * maxRange / 5))
        image = cv2.circle(image,
                           center,
                           5,
                           color=(255,60,60),
                           thickness = 2)
        drawLoc = (center[0] + 10, center[1] - 10)
        image = cv2.putText(image, f'{det[6]:0.2f}', drawLoc, cv2.FONT_HERSHEY_SIMPLEX,  
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
