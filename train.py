import time
import tensorflow as tf
import psutil
from dataset.custom_batch_data_generator import DataGenerator
import util.loader as loader
from train_utils import *
from config_utils import *


def main():

    setup_gpus()

    config_data, config_model, config_train = get_configuration()

    anchor_boxes = loader.readAnchorBoxes()
    num_classes = len(config_data["all_classes"])

    model = create_model(config_model, config_data, config_train, anchor_boxes)

    data_generator = DataGenerator(config_data, config_train, config_model,
                                   model.features_shape, anchor_boxes)
    validate_generator = data_generator.validateGenerator()
    train_generator = data_generator.trainGenerator()

    global_steps, optimizer, writer, manager, ckpt = setup_training_configuration(config_train, model)

    restore_checkpoint(ckpt, manager, global_steps)

    ###---------------------------- TRAIN SET -------------------------###
    for data, label, raw_boxes in train_generator \
            .repeat() \
            .batch(data_generator.batch_size) \
            .take(data_generator.total_train_batches):

        total_loss, box_loss, conf_loss, category_loss = train_step(
            model, optimizer, writer, global_steps, data, label, raw_boxes)

        tf.print(f"=======> train step: {int(global_steps):4d}, \
            lr: {optimizer.lr.numpy():.6f}, \
            total_loss: {total_loss:.2f}, \
            box_loss: {box_loss:.2f}, \
            conf_loss: {conf_loss:.2f}, \
            category_loss: {category_loss:.2f}")

        # custom_step()
        global_steps.assign_add(1)
        calculate_learning_rate(optimizer, global_steps, config_train)

        ###---------------------------- VALIDATE SET -------------------------###
        if global_steps.numpy() >= config_train["validate_start_steps"] and \
           global_steps.numpy() % config_train["validate_gap"] == 0:

            print('memory usesd: ' + str(psutil.virtual_memory().used // 1e6))

            beforeSaving = time.time()
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(
                int(ckpt.step), save_path))

            beforeValidation = time.time()
            mean_ap_test, ap_all_class_test, ap_all_class_new = validate_step(
                model, data_generator, config_model, validate_generator, global_steps, config_train, num_classes)
            afterValidation = time.time()
            print(
                f"Time spent on validation step {afterValidation - beforeValidation}, on saving the checkopoint {beforeValidation - beforeSaving}")


if __name__ == "__main__":
    main()
