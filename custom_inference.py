# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
from hungarian import hungarian_algorithm
import util.drawer as drawer
import util.helper as helper
import util.loader as loader
import metrics.mAP as mAP
from dataset.custom_batch_data_generator import RawDataLoader
from dataset.custom_batch_data_generator import RadarDataLoader
import model.model_cart as MCart
from model.raddet import RADDet
import time
from tqdm import tqdm
from glob import glob
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


def loadDataForPlot(dataLoader, config_data, showSynchronizedVideo,
                    config_radar, interpolation=15.):
    """ Load data one by one for generating evaluation images """
    sequence_num = -1
    totalSamples = dataLoader.get_total_samples()
    # for RAD_file in all_RAD_files:
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
            img_file = "images/calib_target.jpg"  # Just a dummy
            stereo_left_image = loader.readStereoLeft(img_file)

        RAD_data = np.log(np.abs(RAD_complex) + 1e-9)
        RAD_data = (RAD_data - config_data["global_mean_log"]) / \
            config_data["global_variance_log"]
        data = tf.expand_dims(tf.constant(RAD_data, dtype=tf.float32), axis=0)
        yield sequence_num, data, stereo_left_image, RD_img, RA_img, RA_cart_img, gt_instances


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

    anchor_boxes = loader.readAnchorBoxes()  # load anchor boxes with order
    anchor_cart = loader.readAnchorBoxes(anchor_boxes_file="./anchors_cartboxes.txt")
    num_classes = len(config_data["all_classes"])

    ### NOTE: using the yolo head shape out from model for data generator ###
    model = RADDet(config_model, config_data, config_train, anchor_boxes)
    model.build([None] + config_model["input_shape"])
    model.backbone_stage.summary()
    model.summary()

    ### NOTE: building another model for Cartesian Boxes ###
    model_cart = MCart.RADDetCart(config_model, config_data, config_train,
                                  anchor_cart, list(model.backbone_fmp_shape))
    model_cart.build([None] + model.backbone_fmp_shape)
    model_cart.summary()

    ### NOTE: RAD Boxes ckpt ###
    logdir = os.path.join(config_inference["log_dir"],
                          "b_" + str(config_train["batch_size"]) +
                          "lr_" + str(config_train["learningrate_init"]))
    if not os.path.exists(logdir):
        raise ValueError("RAD Boxes model not loaded, please check the ckpt path.")
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config_train["learningrate_init"])
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model, step=global_steps)
    manager = tf.train.CheckpointManager(ckpt,
                                         os.path.join(logdir, "ckpt"), max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored RAD Boxes Model from {}".format(manager.latest_checkpoint))
    else:
        raise ValueError("RAD Boxes model not loaded, please check the ckpt path.")

    ### NOTE: Cartesian Boxes ckpt ###
    if_evaluate_cart = True
    logdir_cart = os.path.join(config_inference["log_dir"], "cartesian_" +
                               "b_" + str(config_train["batch_size"]) +
                               "lr_" + str(config_train["learningrate_init"]))
    # "lr_" + str(config_train["learningrate_init"]) + \
    # "_" + str(config_train["log_cart_add"]))
    if not os.path.exists(logdir_cart):
        if_evaluate_cart = False
        print("*************************************************************")
        print("Cartesian ckpt not found, skipping evaluating Cartesian Boxes")
        print("*************************************************************")
    if if_evaluate_cart:
        global_steps_cart = tf.Variable(1, trainable=False, dtype=tf.int64)
        optimizer_cart = tf.keras.optimizers.Adam(learning_rate=config_train["learningrate_init"])
        ckpt_cart = tf.train.Checkpoint(optimizer=optimizer_cart, model=model_cart,
                                        step=global_steps_cart)
        manager_cart = tf.train.CheckpointManager(ckpt_cart,
                                                  os.path.join(logdir_cart, "ckpt"), max_to_keep=3)
        ckpt_cart.restore(manager_cart.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored Cartesian Boxes Model from {}".format
                  (manager_cart.latest_checkpoint))

    def inferencePlotting(all_original_files, showSynchronizedVideo, calculateErrors):
        """ Plot the predictions of all data in dataset """

        globalEndsList = []  # Last frames for each of the measurement
        globalIndex = 0

        def insert_file_end():
            globalEndsList.append(globalIndex)

        if if_evaluate_cart or showSynchronizedVideo:
            fig, axes = drawer.prepareFigure(3, figsize=(40, 6))
        else:
            fig, axes = drawer.prepareFigure(2)
        colors = loader.randomColors(config_data["all_classes"])

        image_save_dir = "./images/inference_plots/"
        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir)
        else:
            shutil.rmtree(image_save_dir)
            os.makedirs(image_save_dir)
        print("Start plotting, it might take a while...")

        if showSynchronizedVideo:
            testLoader = RadarDataLoader(all_original_files,
                                         config_inference["loader_samples_ratio"],
                                         config_model["input_shape"][2],
                                         2,
                                         samplesToLoad=15000,
                                         randomize=False,
                                         showSynchronizedVideo=showSynchronizedVideo,
                                         finishedFileCallback=insert_file_end)
        else:
            testLoader = RawDataLoader("../Downloads/empty/2022-11-07_09_22_27/2022-11-07_09_22_27_tx2.raw")

        testLoader.logLevel = 3
        totalSamples = testLoader.get_total_samples()
        if not totalSamples:
            print('No data to process. Wrong "original_videos_pattern" value in config.json?')
            return

        pbar = tqdm(total=totalSamples)
        model_RAD_st = []
        model_cart_st = []
        firstIteration = True
        thresholds = list(a for a in np.arange(0.5, 0.95, 0.05))
        print(thresholds)
        #thresholds = [0.8]
        distanceSqErrors = []
        missedDetections = []
        falseDetections = []
        detectedOnEmpty = []
        detectedOnEmptyPerFrame = []
        for i in range(len(thresholds)):
            distanceSqErrors.append([])
            missedDetections.append(0)
            falseDetections.append(0)
            detectedOnEmpty.append(0)
            detectedOnEmptyPerFrame.append(0)
        netResults = {}
        gtResults = {}
        curResultIndex = 0
        for sequence_num, data, stereo_left_image, RD_img, RA_img, RA_cart_img, gt_instances in \
                loadDataForPlot(testLoader, config_data, showSynchronizedVideo, config_radar):

            globalIndex += 1

            if data is None or stereo_left_image is None:
                pbar.update(1)
                continue
            model_RAD_time_start = time.time()

            if showSynchronizedVideo:
                RA_cart_img = stereo_left_image[:, :, ::-1]

            if firstIteration:
                feature = model(data)
                np.save("network_in.npy", data)
                tf.saved_model.save(model, "model_b_" + str(config_train["batch_size"]) +
                                    "lr_" + str(config_train["learningrate_init"]))
                np.save("network_out.npy", feature)
                firstIteration = False
            else:
                feature = modelAsGraph(model, data)
            pred_raw, pred = model.decodeYolo(feature)

            #np.save("decoded2.npy", pred)

            pred_frame = pred[0]
            predicitons = helper.yoloheadToPredictions(pred_frame,
                                                       conf_threshold=config_evaluate["confidence_threshold"])
            print(predicitons)
            print(f"NMS IoU threshold {config_inference['nms_iou3d_threshold']}")
            nms_pred = helper.nms(predicitons,
                                  config_inference["nms_iou3d_threshold"],
                                  config_model["input_shape"],
                                  sigma=0.3, method="nms")

            netResults[curResultIndex] = nms_pred
            gtResults[curResultIndex] = gt_instances["boxes"]
            curResultIndex += 1

            # A custom additional print
            distDisplace = 8
            for d in range(nms_pred.shape[0]):
                cls = int(nms_pred[d, 7])
                print(
                    f"Detected {config_data['all_classes'][cls]} at {int(nms_pred[d, 0])}, {int(nms_pred[d, 1])}, "
                    f"{int(nms_pred[d, 2])} ({int(nms_pred[d, 3])} by {int(nms_pred[d, 4])} by {int(nms_pred[d, 5])}) "
                    f"with prob {nms_pred[d, 6]}")

            if calculateErrors:
                for tIndex in range(len(thresholds)):
                    cartesianDetections = []
                    cartesianGt = []
                    for d in range(nms_pred.shape[0]):
                        if nms_pred[d, 6] >= thresholds[tIndex]:
                            cart = rad_to_cartesian(nms_pred[d, 0] - distDisplace,
                                                    nms_pred[d, 1],
                                                    config_radar["azimuth_size"],
                                                    config_radar["range_resolution"],
                                                    config_radar["angular_resolution"])
                            cartesianDetections.append(cart)

                    #print(f"Ground truths on {sequence_num}:")
                    for box in gt_instances["boxes"]:
                        cart = rad_to_cartesian(box[0],
                                                box[1],
                                                config_radar["azimuth_size"],
                                                config_radar["range_resolution"],
                                                config_radar["angular_resolution"])
                        cartesianGt.append(cart)
                        #print(f"Cartesian {cart} from RA ({box[0]}, {box[1]})")

                    largest = max(len(cartesianGt), len(cartesianDetections))
                    costMatrix = 100 * np.ones((largest, largest))

                    for g in range(len(cartesianGt)):
                        for d in range(len(cartesianDetections)):
                            x1 = cartesianGt[g][0]
                            y1 = cartesianGt[g][1]
                            x2 = cartesianDetections[d][0]
                            y2 = cartesianDetections[d][1]
                            costMatrix[g, d] = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
                    #print(f"Cost matrix:\n{costMatrix}")

                    assignments = hungarian_algorithm(costMatrix.copy())
                    #print(f"For threshold {thresholds[tIndex]} assignments: {assignments}")

                    for c, a in enumerate(assignments):
                        if c >= len(cartesianGt):
                            break  # further entries are due to the matrix augmentation

                        if costMatrix[a[0], a[1]] > 99:  # No assignment found
                            missedDetections[tIndex] += 1
                        else:
                            distanceSqErrors[tIndex].append(costMatrix[a[0], a[1]])

                    if len(cartesianDetections) > len(cartesianGt):
                        #print(f"There are false detections. len(cartesianDetections) {len(cartesianDetections)}, len(cartesianGt) {len(cartesianGt)}")
                        falseDetections[tIndex] += len(cartesianDetections) - len(cartesianGt)
            else:
                for tIndex in range(len(thresholds)):
                    for d in range(nms_pred.shape[0]):
                        if nms_pred[d, 6] > thresholds[tIndex]:
                            detectedOnEmptyPerFrame[tIndex] += 1
                            break
                    for d in range(nms_pred.shape[0]):
                        if nms_pred[d, 6] > thresholds[tIndex]:
                            detectedOnEmpty[tIndex] += 1

            if False:
                with open("plots.txt", 'a') as f:
                    outputString = ""
                    for d in range(nms_pred.shape[0]):
                        outputString += f"{nms_pred[d, 0]}, "
                    if nms_pred.shape[0] < 8:
                        for _ in range(nms_pred.shape[0], 8):
                            outputString += "-1.0, "
                    f.write(outputString+"\n")
            model_RAD_st.append(time.time() - model_RAD_time_start)
            if if_evaluate_cart:
                model_cart_time_start = time.time()
                backbone_fmp = model.backbone_stage(data)
                pred_raw_cart = model_cart(backbone_fmp)
                pred_cart = model_cart.decodeYolo(pred_raw_cart)
                pred_frame_cart = pred_cart[0]
                predicitons_cart = helper.yoloheadToPredictions2D(pred_frame_cart,
                                                                  conf_threshold=0.5)
                nms_pred_cart = helper.nms2D(predicitons_cart,
                                             config_inference["nms_iou3d_threshold"],
                                             config_model["input_shape"],
                                             sigma=0.3, method="nms")
                for d in range(nms_pred_cart.shape[0]):
                    cls = int(nms_pred_cart[d, 5])
                    print(f"Detected {config_data['all_classes'][cls]} at {int(nms_pred_cart[d, 0])}, "
                          f"{int(nms_pred_cart[d, 1])} ({int(nms_pred_cart[d, 2])} by {int(nms_pred_cart[d, 3])})"
                          f" with prob {nms_pred_cart[d, 4]}")

                model_cart_st.append(time.time() - model_cart_time_start)
            else:
                nms_pred_cart = None
            drawer.clearAxes(axes)
            drawer.drawInference(RD_img,
                                 RA_img, RA_cart_img, nms_pred,
                                 config_data["all_classes"], colors, axes,
                                 radar_cart_nms=nms_pred_cart)
            drawer.drawGt(RD_img, RA_img, gt_instances, axes)
            drawer.saveFigure(image_save_dir, "%.6d.png" % (sequence_num))

            if showSynchronizedVideo:
                cutImageCustom(image_save_dir, "%.6d.png" % (sequence_num))
            '''
            # This is the original implementation removed from this logic
            if if_evaluate_cart:
                cutImage(image_save_dir, "%.6d.png"%(sequence_num))
            else:
                cutImage3Axes(image_save_dir, "%.6d.png"%(sequence_num))
            '''
            pbar.update(1)
        print("------", " The average inference time for RAD Boxes: ",
              np.mean(model_RAD_st))
        if if_evaluate_cart:
            print("======", " The average inference time for Cartesian Boxes: ",
                  np.mean(model_cart_st))
        if calculateErrors:
            for tInd in range(len(thresholds)):
                reportName = f"errors_{thresholds[tInd]:.2}.txt"
                with open(reportName, "w") as f:
                    f.write(str(distanceSqErrors[tInd]) + "\n")
                    f.write(f"Missed: {missedDetections[tInd]}\n")
                    f.write(f"False: {falseDetections[tInd]}\n")
                    mse = np.sum(distanceSqErrors[tInd]) / len(distanceSqErrors[tInd])
                    f.write(f"MSE: {mse}\n")
        else:
            print(f"Detected on empty: {detectedOnEmpty}")
            print(f"Detected on empty per frame: {detectedOnEmptyPerFrame}")

        np.save("resultingDetections3.npy", netResults)
        np.save("groundTruths3.npy", gtResults)
        np.save("measurementLastFrames3.npy", globalEndsList)
        print(f"globalEndsList: {globalEndsList}")
    ### NOTE: inference starting from here ###
    all_original_files = config_inference["original_videos_pattern"]
    showSynchronizedVideo = True
    calculateErrors = True
    inferencePlotting(all_original_files, showSynchronizedVideo, calculateErrors)


if __name__ == "__main__":
    main()
