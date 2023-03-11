# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import tensorflow as tf
import numpy as np
import glob, os
import sys

import util.loader as loader
import util.helper as helper

import random
import time

from dataset.RadarDataLoader import RadarDataLoader
from dataset.RawDataLoader import RawDataLoader

class DataGenerator:
    def __init__(self, config_data, config_train, config_model, headoutput_shape, \
                anchors, anchors_cart=None, cart_shape=None):
        """ Data Generator:
            Data, Gt loader and generator, all sequences are based on the file
        PROJECT_ROOT/sequences.txt. 
        """
        self.input_size = config_model["input_shape"]
        self.config_data = config_data
        self.config_train = config_train
        self.config_model = config_model
        self.headoutput_shape = headoutput_shape
        self.cart_shape = cart_shape
        self.grid_strides = self.getGridStrides()
        self.cart_grid_strides = self.getCartGridStrides()
        self.anchor_boxes = anchors
        self.anchor_boxes_cart = anchors_cart


        trainPattern = config_train["original_videos_pattern_train"]
        valPattern = config_train["original_videos_pattern_val"]
        self.trainLoader = RadarDataLoader(trainPattern, 0.02, config_model["input_shape"][2], 2, samplesToLoad=25000)
        self.valLoader = RadarDataLoader(valPattern, 0.005, config_model["input_shape"][2], 2, samplesToLoad=5000, randomize=False)
        self.trainLoader.set_augmentation_probabilities(flipProb = 0.0)

        # Dummy
        self.testLoader = RadarDataLoader("D:/TPP_val/*/*_wall.avi", 0.005, config_model["input_shape"][2], 2, samplesToLoad=5000)

        self.trainLoader.logLevel = 1
        self.valLoader.logLevel = 0
        self.testLoader.logLevel = 0

        totalTrainSamples = self.trainLoader.get_total_samples()
        totalValSamples = self.valLoader.get_total_samples()
        totalTestSamples = self.valLoader.get_total_samples()
        print(f"Total number of samples to received from the train data loader {totalTrainSamples}")
        print(f"Total number of samples to received from the val data loader {totalValSamples}")

        self.batch_size = config_train["batch_size"]
        self.total_train_batches = (self.config_train["epochs"] * totalTrainSamples) // self.batch_size
        self.total_test_batches = totalTestSamples // self.batch_size
        self.total_validate_batches = totalValSamples // self.batch_size

    def getGridStrides(self, ):
        """ Get grid strides """
        strides = (np.array(self.config_model["input_shape"])[:3] / \
                            np.array(self.headoutput_shape[1:4]))
        return np.array(strides).astype(np.float32)

    """---------------------------------------------------------------------"""
    """-------------------- RAD 3D Boxes train/test set --------------------"""
    """---------------------------------------------------------------------"""
    def encodeToLabels(self, gt_instances):
        """ Transfer ground truth instances into Detection Head format """
        raw_boxes_xyzwhd = np.zeros((self.config_data["max_boxes_per_frame"], 7))
        ### initialize gronud truth labels as np.zeors ###
        gt_labels = np.zeros(list(self.headoutput_shape[1:4]) + \
                        [len(self.anchor_boxes)] + \
                        [len(self.config_data["all_classes"]) + 7])

        ### start transferring box to ground turth label format ###
        for i in range(len(gt_instances["classes"])):
            if i > self.config_data["max_boxes_per_frame"]:
                continue
            class_name = gt_instances["classes"][i]
            box_xyzwhd = gt_instances["boxes"][i]
            
            class_id = self.config_data["all_classes"].index(class_name)
            if i < self.config_data["max_boxes_per_frame"]:
                raw_boxes_xyzwhd[i, :6] = box_xyzwhd
                raw_boxes_xyzwhd[i, 6] = class_id
            class_onehot = helper.smoothOnehot(class_id, len(self.config_data["all_classes"]))
            
            exist_positive = False

            grid_strid = self.grid_strides
            anchor_stage = self.anchor_boxes
            box_xyzwhd_scaled = box_xyzwhd[np.newaxis, :].astype(np.float32)
            box_xyzwhd_scaled[:, :3] /= grid_strid
            anchorstage_xyzwhd = np.zeros([len(anchor_stage), 6])
            anchorstage_xyzwhd[:, :3] = np.floor(box_xyzwhd_scaled[:, :3]) + 0.5
            anchorstage_xyzwhd[:, 3:] = anchor_stage.astype(np.float32)

            iou_scaled = helper.iou3d(box_xyzwhd_scaled, anchorstage_xyzwhd, \
                                        self.input_size)
            ### NOTE: 0.3 is from YOLOv4, maybe this should be different here ###
            ### it means, as long as iou is over 0.3 with an anchor, the anchor
            ### should be taken into consideration as a ground truth label
            iou_mask = iou_scaled > 0.3

            if np.any(iou_mask):
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled)[:3]).\
                                    astype(np.int32)
                ### TODO: consider changing the box to raw yolohead output format ###
                gt_labels[xind, yind, zind, iou_mask, 0:6] = box_xyzwhd
                gt_labels[xind, yind, zind, iou_mask, 6:7] = 1.
                gt_labels[xind, yind, zind, iou_mask, 7:] = class_onehot
                exist_positive = True

            if not exist_positive:
                ### NOTE: this is the normal one ###
                ### it means take the anchor box with maximum iou to the raw
                ### box as the ground truth label
                anchor_ind = np.argmax(iou_scaled)
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled)[:3]).\
                                    astype(np.int32)
                gt_labels[xind, yind, zind, anchor_ind, 0:6] = box_xyzwhd
                gt_labels[xind, yind, zind, anchor_ind, 6:7] = 1.
                gt_labels[xind, yind, zind, anchor_ind, 7:] = class_onehot

        has_label = False
        for label_stage in gt_labels:
            if label_stage.max() != 0:
                has_label = True
                
        gt_labels = [np.where(gt_i == 0, 1e-16, gt_i) for gt_i in gt_labels]
        return gt_labels, has_label, raw_boxes_xyzwhd

    def trainData(self,):
        """ Generate train data with batch size """
        count = 0
        while True:
            frame, gt_instances = next(self.trainLoader.yield_next_data())
            ### NOTE: Gloabl Normalization ###
            RAD_data = np.log(np.abs(frame) + 1e-9) # Added a small to limit the scale of a logarithm
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / \
                                self.config_data["global_variance_log"]

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)

            #if has_label:
            yield (RAD_data, gt_labels, raw_boxes)
     
    def testData(self, ):
        """ Generate train data with batch size """
        count = 0
        while True:
            frame, gt_instances = next(self.testLoader.yield_next_data())

            ### NOTE: Gloabl Normalization ###
            RAD_data = np.log(np.abs(frame) + 1e-9) # Added a small to limit the scale of a logarithm
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / \
                                self.config_data["global_variance_log"]

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)

            #if has_label:
            yield (RAD_data, gt_labels, raw_boxes)
            
    def validateData(self, ):
        """ Generate train data with batch size """
        count = 0
        while True:
            frame, gt_instances = next(self.valLoader.yield_next_data())

            ### NOTE: Gloabl Normalization ###
            RAD_data = np.log(np.abs(frame) + 1e-9) # Added a small to limit the scale of a logarithm
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / \
                                self.config_data["global_variance_log"]

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)

            #if has_label:
            yield (RAD_data, gt_labels, raw_boxes)

    def trainGenerator(self,):
        """ Building data generator using tf.data.Dataset.from_generator """
        return tf.data.Dataset.from_generator(self.trainData, \
                    output_types=(tf.float32, tf.float32, tf.float32), \
                    output_shapes=(tf.TensorShape(self.config_model["input_shape"]), \
                            tf.TensorShape(list(self.headoutput_shape[1:4]) + \
                            [len(self.anchor_boxes), \
                            7+len(self.config_data["all_classes"])]), \
                            tf.TensorShape([self.config_data["max_boxes_per_frame"], 7]) \
                            ), )


    def testGenerator(self,):
        """ Building data generator using tf.data.Dataset.from_generator """
        return tf.data.Dataset.from_generator(self.testData, \
                    output_types=(tf.float32, tf.float32, tf.float32), \
                    output_shapes=(tf.TensorShape(self.config_model["input_shape"]), \
                            tf.TensorShape(list(self.headoutput_shape[1:4]) + \
                            [len(self.anchor_boxes), \
                            7+len(self.config_data["all_classes"])]), \
                            tf.TensorShape([self.config_data["max_boxes_per_frame"], 7]) \
                            ), )
 
    def validateGenerator(self,):
        """ Building data generator using tf.data.Dataset.from_generator """
        return tf.data.Dataset.from_generator(self.validateData, \
                    output_types=(tf.float32, tf.float32, tf.float32), \
                    output_shapes=(tf.TensorShape(self.config_model["input_shape"]), \
                            tf.TensorShape(list(self.headoutput_shape[1:4]) + \
                            [len(self.anchor_boxes), \
                            7+len(self.config_data["all_classes"])]), \
                            tf.TensorShape([self.config_data["max_boxes_per_frame"], 7]) \
                            ), )

    """---------------------------------------------------------------------"""
    """----------------- Cartesian 2D Boxes train/test set -----------------"""
    """---------------------------------------------------------------------"""
    def getCartGridStrides(self, ):
        """ Get grid strides """
        if self.cart_shape is not None:
            cart_output_shape = [int(self.config_model["input_shape"][0]), \
                                int(2 * self.config_model["input_shape"][0])]
            strides = (np.array(cart_output_shape) / np.array(self.cart_shape[1:3]))
            return np.array(strides).astype(np.float32)
        else:
            return None

    def encodeToCartBoxesLabels(self, gt_instances):
        """ Transfer ground truth instances into Detection Head format """
        raw_boxes_xywh = np.zeros((self.config_data["max_boxes_per_frame"], 5))
        ### initialize gronud truth labels as np.zeros ###
        gt_labels = np.zeros(list(self.cart_shape[1:3]) + \
                        [len(self.anchor_boxes_cart)] + \
                        [len(self.config_data["all_classes"]) + 5]) 

        ### start transferring box to ground turth label format ###
        for i in range(len(gt_instances["classes"])):
            if i > self.config_data["max_boxes_per_frame"]:
                continue
            class_name = gt_instances["classes"][i]
            box_xywh = gt_instances["cart_boxes"][i]
            class_id = self.config_data["all_classes"].index(class_name)
            if i <= self.config_data["max_boxes_per_frame"]:
                raw_boxes_xywh[i, :4] = box_xywh
                raw_boxes_xywh[i, 4] = class_id
            class_onehot = helper.smoothOnehot(class_id, \
                                    len(self.config_data["all_classes"]))
            exist_positive = False
            grid_strid = self.cart_grid_strides
            anchors = self.anchor_boxes_cart
            box_xywh_scaled = box_xywh[np.newaxis, :].astype(np.float32)
            box_xywh_scaled[:, :2] /= grid_strid
            anchors_xywh = np.zeros([len(anchors), 4])
            anchors_xywh[:, :2] = np.floor(box_xywh_scaled[:, :2]) + 0.5
            anchors_xywh[:, 2:] = anchors.astype(np.float32)

            iou_scaled = helper.iou2d(box_xywh_scaled, anchors_xywh)
            ### NOTE: 0.3 is from YOLOv4, maybe this should be different here ###
            ### it means, as long as iou is over 0.3 with an anchor, the anchor
            ### should be taken into consideration as a ground truth label
            iou_mask = iou_scaled > 0.3

            if np.any(iou_mask):
                xind, yind = np.floor(np.squeeze(box_xywh_scaled)[:2]).astype(np.int32)
                ### TODO: consider changing the box to raw yolohead output format ###
                gt_labels[xind, yind, iou_mask, 0:4] = box_xywh
                gt_labels[xind, yind, iou_mask, 4:5] = 1.
                gt_labels[xind, yind, iou_mask, 5:] = class_onehot
                exist_positive = True

            if not exist_positive:
                ### NOTE: this is the normal one ###
                ### it means take the anchor box with maximum iou to the raw
                ### box as the ground truth label
                iou_mask = iou_scaled == iou_scaled.max()

                if np.any(iou_mask):
                    xind, yind = np.floor(np.squeeze(box_xywh_scaled)[:2]).astype(np.int32)
                    ### TODO: consider changing the box to raw yolohead output format ###
                    gt_labels[xind, yind, iou_mask, 0:4] = box_xywh
                    gt_labels[xind, yind, iou_mask, 4:5] = 1.
                    gt_labels[xind, yind, iou_mask, 5:] = class_onehot

        has_label = False
        if gt_labels.max() != 0:
            has_label = True
        gt_labels = np.where(gt_labels == 0, 1e-16, gt_labels)
        return gt_labels, has_label, raw_boxes_xywh
    
    def trainDataCart(self,):
        """ Generate train data with batch size """
        if self.cart_grid_strides is None:
            raise ValueError("Cartesian grid is None, please double check")
        count = 0
        while  count < len(self.RAD_sequences_train):
            RAD_filename = self.RAD_sequences_train[count] 
            RAD_complex = loader.readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            RAD_data = helper.complexTo2Channels(RAD_complex)
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / \
                                self.config_data["global_variance_log"]
            ### load ground truth instances ###
            gt_filename = loader.gtfileFromRADfile(RAD_filename, \
                                        self.config_data["train_set_dir"])
            gt_instances = loader.readRadarInstances(gt_filename)
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToCartBoxesLabelsCustom(gt_instances)

            if has_label:
                yield (RAD_data, gt_labels, raw_boxes)
            count += 1
            if count == len(self.RAD_sequences_train) - 1:
                # np.random.seed() # should I add seed here ?
                np.random.shuffle(self.RAD_sequences_train)

    def testDataCart(self, ):
        if self.cart_grid_strides is None:
            raise ValueError("Cartesian grid is None, please double check")
        """ Generate test data with batch size """
        count = 0
        while  count < len(self.RAD_sequences_test):
            RAD_filename = self.RAD_sequences_test[count] 
            RAD_complex = loader.readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            RAD_data = helper.complexTo2Channels(RAD_complex)
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / \
                                self.config_data["global_variance_log"]
            ### load ground truth instances ###
            gt_filename = loader.gtfileFromRADfile(RAD_filename, \
                                        self.config_data["test_set_dir"])
            gt_instances = loader.readRadarInstances(gt_filename)
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToCartBoxesLabels(gt_instances)

            if has_label:
                yield (RAD_data, gt_labels, raw_boxes)
            count += 1

    def validateDataCart(self, ):
        if self.cart_grid_strides is None:
            raise ValueError("Cartesian grid is None, please double check")
        """ Generate test data with batch size """
        count = 0
        while  count < len(self.RAD_sequences_validate):
            RAD_filename = self.RAD_sequences_validate[count] 
            RAD_complex = loader.readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            RAD_data = helper.complexTo2Channels(RAD_complex)
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / \
                                self.config_data["global_variance_log"]
            ### load ground truth instances ###
            gt_filename = loader.gtfileFromRADfile(RAD_filename, \
                                        self.config_data["train_set_dir"])
            gt_instances = loader.readRadarInstances(gt_filename)
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToCartBoxesLabels(gt_instances)

            if has_label:
                yield (RAD_data, gt_labels, raw_boxes)
            count += 1

    def trainCartGenerator(self,):
        """ Building data generator using tf.data.Dataset.from_generator """
        return tf.data.Dataset.from_generator(self.trainDataCart, \
                    output_types=(tf.float32, tf.float32, tf.float32), \
                    output_shapes=(tf.TensorShape(self.config_model["input_shape"]), \
                            tf.TensorShape(list(self.cart_shape[1:3]) + \
                            [len(self.anchor_boxes_cart)] + \
                            [len(self.config_data["all_classes"]) + 5]), 
                            tf.TensorShape([self.config_data["max_boxes_per_frame"], 5]) \
                            ), )


    def testCartGenerator(self,):
        """ Building data generator using tf.data.Dataset.from_generator """
        return tf.data.Dataset.from_generator(self.testDataCart, \
                    output_types=(tf.float32, tf.float32, tf.float32), \
                    output_shapes=(tf.TensorShape(self.config_model["input_shape"]), \
                            tf.TensorShape(list(self.cart_shape[1:3]) + \
                            [len(self.anchor_boxes_cart)] + \
                            [len(self.config_data["all_classes"]) + 5]), 
                            tf.TensorShape([self.config_data["max_boxes_per_frame"], 5]) \
                            ), )

    def validateCartGenerator(self,):
        """ Building data generator using tf.data.Dataset.from_generator """
        return tf.data.Dataset.from_generator(self.validateDataCart, \
                    output_types=(tf.float32, tf.float32, tf.float32), \
                    output_shapes=(tf.TensorShape(self.config_model["input_shape"]), \
                            tf.TensorShape(list(self.cart_shape[1:3]) + \
                            [len(self.anchor_boxes_cart)] + \
                            [len(self.config_data["all_classes"]) + 5]), 
                            tf.TensorShape([self.config_data["max_boxes_per_frame"], 5]) \
                            ), )
