# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import tensorflow as tf
import model.layers as L


def basicResidualBlock(x, channel_expansion, strides=(1, 1), use_bias=False):
    """ Basic block of resnet3d """
    assert isinstance(strides, tuple)
    input_channel = x.shape[-1]
    conv = L.convolution2D(x, input_channel, 3,
                           strides, "same", "relu", use_bias=use_bias, bn=True)
    conv = L.convolution2D(conv, int(input_channel*channel_expansion), 3,
                           (1, 1), "same", "relu", use_bias=use_bias, bn=True)
    conv = L.convolution2D(conv, conv.shape[-1], 1,
                           (1, 1), "same", "relu", use_bias=use_bias, bn=True)
    ### shortcut for the original input ###
    if any(val != 1 for val in strides) or channel_expansion != 1:
        conv_shortcut = L.convolution2D(x, int(input_channel*channel_expansion), 3,
                                        strides, "same", "relu", use_bias=use_bias, bn=True)
    else:
        conv_shortcut = x
    ### combine all together to generate residual feature maps ###
    conv = tf.keras.layers.Add()([conv, conv_shortcut])
    return conv


def bottleneckResidualBlock(x, channel_expansion, strides=(1, 1), use_bias=False):
    """ Basic block of resnet3d """
    assert isinstance(strides, tuple)
    input_channel = x.shape[-1]
    conv = L.convolution2D(x, input_channel, 1,
                           (1, 1), "same", "relu", use_bias=use_bias, bn=True)
    conv = L.convolution2D(x, input_channel, 3,
                           strides, "same", "relu", use_bias=use_bias, bn=True)
    conv = L.convolution2D(conv, int(input_channel*channel_expansion), 3,
                           (1, 1), "same", "relu", use_bias=use_bias, bn=True)
    conv = L.convolution2D(conv, conv.shape[-1], 1,
                           (1, 1), "same", "relu", use_bias=use_bias, bn=True)
    ### shortcut for the original input ###
    if any(val != 1 for val in strides) or channel_expansion != 1:
        conv_shortcut = L.convolution2D(x, int(input_channel*channel_expansion), 3,
                                        strides, "same", "relu", use_bias=use_bias, bn=True)
    else:
        conv_shortcut = x
    ### combine all together to generate residual feature maps ###
    conv = tf.keras.layers.Add()([conv, conv_shortcut])
    return conv


def repeatBlock(conv, repeat_times, all_strides=None, all_expansions=None,
                feature_maps_downsample=False):
    """ Iterative block """
    """ 
    It contains: 1. repeated residual blocks; 
                    2. channels upsample block; 
                    3. feature map size upsample block 
    """
    ###### Assert if the parameters are the ones required ######
    if all_strides is not None and all_expansions is not None:
        assert (isinstance(all_strides, tuple) or isinstance(all_strides, list))
        assert (isinstance(all_expansions, tuple) or isinstance(all_expansions, list))
        assert len(all_strides) == repeat_times
        assert len(all_expansions) == repeat_times
    elif all_strides is None and all_expansions is None:
        all_strides, all_expansions = [], []
        for i in range(repeat_times):
            all_strides.append(1)
            if i % 2 == 0:
                all_expansions.append(0.5)
            else:
                all_expansions.append(2)
    ##### Start repeated block #####
    for i in range(repeat_times):
        strides = (all_strides[i], all_strides[i])
        expansion = all_expansions[i]
        conv = basicResidualBlock(conv, expansion, strides, use_bias=True)
    if feature_maps_downsample:
        conv = L.maxPooling2D(conv)
    return conv


def radarResNet3D(x):
    """
    Builds a 3D ResNet backbone for radar data classification.
    Args:
        x: input tensor

    Returns:
        features: output tensor after processing through the ResNet backbone
    """

    # Set up initial convolution layer
    conv = x

    # Set up block parameters
    block_repeat_times = [2, 2, 4, 8]
    channels_upsample = [False, False, True, True]
    feature_mp_downsample = [True, True, False, False]
    feature_stages = []

    # Repeated residual blocks
    for i, repeat_times in enumerate(block_repeat_times):
        all_strides, all_expansions = [1] * repeat_times, [1] * repeat_times

        # Set stride and expansion factors for each block
        if repeat_times != 1:
            all_strides = [1, 1] * int(repeat_times / 2)
            all_expansions = [1, 1] * int(repeat_times / 2)
        if channels_upsample[i]:
            all_expansions[-1] *= 2

        # Perform the repeated block operations
        feature_maps_downsample = feature_mp_downsample[i]
        conv = repeatBlock(conv, repeat_times, all_strides, all_expansions, feature_maps_downsample)

        # Append the feature stage to a list
        if i > len(block_repeat_times) - 4:
            feature_stages.append(conv)

    # Print the shape of each feature stage
    for stage_i in feature_stages:
        print("--- backbone stage shape ---", stage_i.shape)

    # Return the last feature stage
    features = feature_stages[-1]
    return features
