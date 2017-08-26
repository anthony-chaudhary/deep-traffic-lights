
import tensorflow as tf
import tensorflow.contrib.slim as slim
from hyperparameters import *


def load_vgg(sess, vgg_path):

    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    
    # TODO  confirm these names are right
    # Using instead of conv4_3

    images = tf.get_default_graph().get_tensor_by_name('image_input:0')
    conv4_3 = tf.get_default_graph().get_tensor_by_name('layer4_out:0')
    keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')

    return images, conv4_3, keep_prob


def prediction_and_location(net, layer_id, Predictions, Locations):

    with tf.variable_scope('prediction_and_location'+layer_id):
        prediction = slim.conv2d(net, NUMBER_PREDICTIONS, [3,3], 
                                 activation_fn=None, scope='prediction')
        prediction = tf.contrib.layers.flatten(prediction)

        location = slim.conv2d(net, NUMBER_LOCATIONS, [3,3], 
                               activation_fn=None, scope='location')
        location = tf.contrib.layers.flatten(location)
        print(location)

    Predictions.append(prediction)
    Locations.append(location)

    return Predictions, Locations


def ssd_layers(conv4_3, number_of_classes):

    Predictions, Locations = [], []

    with slim.arg_scope([slim.conv2d]):

        Predictions, Locations = prediction_and_location(conv4_3, 'vgg_0', Predictions, Locations)

        net = slim.conv2d(conv4_3, 1024, [3,3], scope='ssd_0')
        net = slim.conv2d(net, 1024, [1,1], scope='ssd_1')

        Predictions, Locations = prediction_and_location(net, 'ssd_1', Predictions, Locations)

        net = slim.conv2d(net, 256, [1,1], scope='ssd_2')
        net = slim.conv2d(net, 512, [3,3], 2, scope='ssd_2_s2')
        
        Predictions, Locations = prediction_and_location(net, 'ssd_2_s2', Predictions, Locations)

        net = slim.conv2d(net, 128, [1,1], scope='ssd_3')
        net = slim.conv2d(net, 256, [3,3], 2, scope='ssd_3_s2')
        
        Predictions, Locations = prediction_and_location(net, 'ssd_3_s2', Predictions, Locations)

    predictions_all = tf.concat(Predictions, 1)
    predictions_locations_all = tf.concat(Locations, 1)

    return predictions_all, predictions_locations_all


def loss_function(predictions_all, predictions_locations_all):

    feature_map_number = 0
    
    for f in FEATURE_MAP_SIZES:
        feature_map_number += f[0] * f[1] * NUMBER_DEFAULT_BOXES
    
    number_total_locations = feature_map_number * 4
    #number_total_predictions = feature_map_number * NUMBER_CLASSES
 
    true_predictions = tf.placeholder(tf.int32, [None, feature_map_number], name="true_predictions")
    true_locations = tf.placeholder(tf.float32, [None, number_total_locations], name="true_locations")
    prediction_loss_mask = tf.placeholder(tf.float32, [None, feature_map_number], name="prediction_mask")

    scores = tf.reshape(predictions_all, [-1, feature_map_number, NUMBER_CLASSES])
    
    prediction_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=true_predictions)
    prediction_loss *= prediction_loss_mask
    prediction_loss = tf.reduce_sum(prediction_loss)

    location_difference = true_locations - predictions_locations_all

    location_loss_l2 = .5 * (pow(location_difference, 2))
    location_loss_l1 = tf.abs(location_difference) - .5
    smooth_l1 = tf.less(tf.abs(location_difference), 1.0)
    location_loss = tf.where(smooth_l1, location_loss_l2, location_loss_l1)

    location_loss_mask = tf.minimum(true_predictions, 1)
    location_loss_mask = tf.to_float(location_loss_mask)
    location_loss_mask = tf.stack([location_loss_mask] * 4, axis=2)
    location_loss_mask = tf.reshape(location_loss_mask, [-1, number_total_locations])
    
    location_loss *= location_loss_mask

    loss = prediction_loss + location_loss + tf.reduce_sum(tf.losses.get_regularization_losses())

    all_probabilities = tf.nn.softmax(scores)
    top_k_probabilities, top_k_prediction_probabilities = tf.nn.top_k(all_probabilities)
    top_k_probabilities = tf.reshape(top_k_probabilities, [-1, feature_map_number])
    top_k_prediction_probabilities = tf.reshape(top_k_prediction_probabilities, [-1, feature_map_number])

    return loss, scores, true_predictions, true_locations, prediction_loss_mask



def optimizer(loss):

    adam = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    return adam

