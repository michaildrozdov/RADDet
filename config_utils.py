import tensorflow as tf
import util.loader as loader
from model.raddet import RADDet
import os


def setup_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.debugging.enable_check_numerics()
    if len(gpus) > 0:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


def get_configuration():
    """
    Get config of the data, model, and training
    """
    config = loader.readConfig()
    return config["DATA"], config["MODEL"], config["TRAIN"]


def create_model(config_model, config_data, config_train, anchor_boxes):
    """Create a model with the specified configuration

    Args:
        config_model (dict): model config
        config_data (dict): data config
        config_train (dict): train config
        anchor_boxes (dict): anchor boxes

    Returns:
        RADDet: built model
    """
    model = RADDet(config_model, config_data, config_train, anchor_boxes)
    model.build([None] + config_model["input_shape"])
    model.summary()
    return model


def setup_training_configuration(config_train, model):
    """Setup optimizer and bookkeeping for training

    Args:
        config_train (dict): train config
        model (RADDet): model
    """
    logdir = os.path.join(config_train["log_dir"],
                          "b_" + str(config_train["batch_size"]) +
                          "lr_" + str(config_train["learningrate_init"]))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config_train["learningrate_init"])
    writer = tf.summary.create_file_writer(logdir)
    ckpt = tf.train.Checkpoint(
        optimizer=optimizer, model=model, step=global_steps)
    log_specific_dir = os.path.join(logdir, "ckpt")
    tf.summary.trace_on(graph=True)
    manager = tf.train.CheckpointManager(ckpt, log_specific_dir, max_to_keep=3)
    return global_steps, optimizer, writer, manager, ckpt


def restore_checkpoint(ckpt, manager, global_steps):
    """
    Restore checkpoint and current step iv possible
    """
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        global_steps.assign(ckpt.step.numpy())
