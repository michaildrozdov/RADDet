import time
import numpy as np
import tensorflow as tf
import util.helper as helper
import metrics.mAP as mAP


def calculate_learning_rate(optimizer, global_steps, config_train):
    ### NOTE: learning rate decay ###
    if global_steps < config_train["warmup_steps"]:
        # lr = config_train["learningrate_init"]
        if global_steps < config_train["startup_steps"]:
            lr = config_train["learningrate_startup"]
        else:
            lr = config_train["learningrate_init"]
        optimizer.lr.assign(lr)
    elif global_steps % config_train["learningrate_decay_gap"] == 0:
        lr = optimizer.lr.numpy()
        lr = config_train["learningrate_end"] + \
            config_train["learningrate_decay"] * \
            (lr - config_train["learningrate_end"])
        optimizer.lr.assign(lr)


@tf.function
def train_step(model, optimizer, writer, global_steps, data, label, raw_boxes):
    """ define train step for training """
    with tf.GradientTape() as tape:
        timeBeforeModel = time.time()
        feature = model(data)
        timeBeforeDecoding = time.time()
        pred_raw, pred = model.decodeYolo(feature)
        timeBeforeLoss = time.time()
        total_loss, box_loss, conf_loss, category_loss = \
            model.loss(pred_raw, pred, label, raw_boxes[..., :6])
        gradients = tape.gradient(total_loss, model.trainable_variables)

        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
        ### NOTE: writing summary data ###
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss",
                              total_loss, step=global_steps)
            tf.summary.scalar("loss/box_loss", box_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss",
                              conf_loss, step=global_steps)
            tf.summary.scalar("loss/category_loss",
                              category_loss, step=global_steps)
        writer.flush()
        timeAfterAll = time.time()
        modelDelta = timeBeforeDecoding - timeBeforeModel
        decodingDelta = timeBeforeLoss - timeBeforeDecoding
        remainingDelta = timeAfterAll - timeBeforeLoss
        print(
            f"Timings: model {modelDelta}, decoding {decodingDelta}, the rest {remainingDelta}")
    return total_loss, box_loss, conf_loss, category_loss


@tf.function
def pred_only(model, data):
    feature = model(data, training=False)
    pred_raw, pred = model.decodeYolo(feature)
    return pred_raw, pred


@tf.function
def validate_step(model, data_generator, config_model, validate_generator, global_steps, config_train, num_classes):
    print(f"Validating on {data_generator.total_validate_batches} batches")
    mean_ap_test = 0.0
    ap_all_class_test = []
    ap_all_class = []
    # total_losstest = []
    # box_losstest = []
    # conf_losstest = []
    # category_losstest = []

    tp_each_class = []
    gt_counts = []
    confidences_each_class = []
    for _ in range(num_classes):
        ap_all_class.append([])
        tp_each_class.append(np.zeros(0))
        confidences_each_class.append(np.zeros(0))
        gt_counts.append(0)
    someIndex = 0
    for data, label, raw_boxes in validate_generator.batch(
            data_generator.batch_size).take(
            data_generator.total_validate_batches):
        someIndex += 1

        pred_raw, pred = pred_only(data)

        # total_loss_b, box_loss_b, conf_loss_b, category_loss_b = model.loss(pred_raw, pred, label, raw_boxes[..., :6])

        # total_losstest.append(total_loss_b)
        # box_losstest.append(box_loss_b)
        # conf_losstest.append(conf_loss_b)
        # category_losstest.append(category_loss_b)

        for batch_id in range(raw_boxes.shape[0]):
            raw_boxes_frame = raw_boxes[batch_id]
            pred_frame = pred[batch_id]
            predictions = helper.yoloheadToPredictions(pred_frame,
                                                       conf_threshold=config_model["confidence_threshold"])
            nms_pred = helper.nms(predictions, config_model["nms_iou3d_threshold"],
                                  config_model["input_shape"], sigma=0.3, method="nms")
            mean_ap, ap_all_class = mAP.mAP(nms_pred, raw_boxes_frame.numpy(),
                                            config_model["input_shape"], ap_all_class,
                                            tp_iou_threshold=config_model["mAP_iou3d_threshold"])
            tp_each_class, gt_counts, confidences_each_class = mAP.appendTp(
                nms_pred, raw_boxes_frame.numpy(),
                config_model["input_shape"],
                tp_each_class, gt_counts, confidences_each_class,
                tp_iou_threshold=config_model["mAP_iou3d_threshold"])
            if not np.isnan(mean_ap):
                mean_ap_test += mean_ap

        if not (someIndex % 10):
            print(f"While validating processed {someIndex}")

    mean_ap_new, ap_all_class_new = mAP.mAPFromAccumulated(
        tp_each_class, gt_counts, confidences_each_class)
    #tf.print("All true positives:")
    # tf.print(tp_each_class)

    for ap_class_i in ap_all_class:
        if len(ap_class_i) == 0:
            class_ap = 0.
        else:
            class_ap = np.mean(ap_class_i)
        ap_all_class_test.append(class_ap)
    mean_ap_test /= data_generator.batch_size*data_generator.total_validate_batches

    tf.print(f"-------> ap: {mean_ap_test:.6f}")
    tf.print(f"-------> ap person: {ap_all_class_test[0]:.6f}")
    tf.print(f"-------> ap bicycle: {ap_all_class_test[1]:.6f}")

    tf.print(f"-------> ap (new): {mean_ap_new:.6f}")
    tf.print(f"-------> ap person (new): {ap_all_class_new[0]:.6f}")
    if len(ap_all_class_new) > 1:
        tf.print(f"-------> ap bicycle (new): {ap_all_class_new[1]:.6f}")

    with open("val_results_" + "b_" + str(config_train["batch_size"]) +
              "lr_" + str(config_train["learningrate_init"]) + ".txt", 'a') as f:
        f.write(f"{float(global_steps)}, {mean_ap_test}, {mean_ap_new},\n")
    ### writing summary data ###
    # with writer.as_default():
    #     tf.summary.scalar("ap/ap_all", mean_ap_test, step=global_steps)
    #     tf.summary.scalar("ap/ap_person", ap_all_class_test[0], step=global_steps)
    #     #tf.summary.scalar("ap/ap_bicycle", ap_all_class_test[1], step=global_steps)
    #     #tf.summary.scalar("ap/ap_car", ap_all_class_test[2], step=global_steps)
    #     #tf.summary.scalar("ap/ap_motorcycle", ap_all_class_test[3], step=global_steps)
    #     #tf.summary.scalar("ap/ap_bus", ap_all_class_test[4], step=global_steps)
    #     #tf.summary.scalar("ap/ap_truck", ap_all_class_test[5], step=global_steps)
    #     ### NOTE: validate loss ###
    #     tf.summary.scalar("validate_loss/total_loss",
    #                       np.mean(total_losstest), step=global_steps)
    #     tf.summary.scalar("validate_loss/box_loss",
    #                       np.mean(box_losstest), step=global_steps)
    #     tf.summary.scalar("validate_loss/conf_loss",
    #                       np.mean(conf_losstest), step=global_steps)
    #     tf.summary.scalar("validate_loss/category_loss",
    #                       np.mean(category_losstest), step=global_steps)
    # writer.flush()
    return mean_ap_test, ap_all_class_test, ap_all_class_new
