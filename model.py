import tensorflow as tf
import tensorflow.contrib.slim as slim
from hyperparameters import *

def load_vgg(sess, vgg_path):

    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    images = tf.get_default_graph().get_tensor_by_name('image_input:0')
    conv4_3_pool = tf.get_default_graph().get_tensor_by_name('pool4:0')
    conv4_3_relu = tf.get_default_graph().get_tensor_by_name('conv4_3/Relu:0')   
    keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')

    return images, conv4_3_pool, conv4_3_relu, keep_prob


def confidences_and_locations(net, layer_id, Confidences, Locations):

    with tf.variable_scope('prediction_and_location_'+layer_id):
        
        num_anchors = (NUMBER_DEFAULT_BOXES)
        prediction = slim.conv2d(net, num_anchors*NUMBER_CLASSES, [3,3], 
                                 activation_fn=None, scope='prediction', padding='SAME')
        prediction = tf.contrib.layers.flatten(prediction)
        location = slim.conv2d(net, num_anchors*4, [3,3], activation_fn=None, scope='location')
        location = tf.contrib.layers.flatten(location)

        Confidences.append(prediction)
        Locations.append(location)

        return Confidences, Locations


def ssd_layers(conv4_3, conv5_pool):

    Confidences, Locations = [], []

    Confidences, Locations = confidences_and_locations(conv5_pool, 'ssd_0_vgg_', Confidences, Locations)

    with tf.variable_scope("ssd_300"):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, 
                            weights_regularizer=slim.l2_regularizer(1e-3), padding='SAME'):


            net = slim.conv2d(conv4_3, 1024, [3,3], scope='ssd_0')
            net = slim.conv2d(net, 1024, [1,1], scope='ssd_1')

            Confidences, Locations = confidences_and_locations(net, 'ssd_1', Confidences, Locations)

            net = slim.conv2d(net, 256, [1,1], scope='ssd_2')
            net = slim.conv2d(net, 512, [3,3], 2, scope='ssd_2_s2')
        
            Confidences, Locations = confidences_and_locations(net, 'ssd_2_s2', Confidences, Locations)

            net = slim.conv2d(net, 128, [1,1], scope='ssd_3')
            net = slim.conv2d(net, 256, [3,3], 2, scope='ssd_3_s2')

            Confidences, Locations = confidences_and_locations(net, 'ssd_3_s2', Confidences, Locations)
        
        final_Confidences = tf.concat(Confidences, 1)
        final_Locations = tf.concat(Locations, 1)
        
        return final_Confidences, final_Locations


def loss_function(confidences_all, locations_all):

    true_confidences = tf.placeholder(tf.int32, [None, NUMBER_CONFIDENCES], name="true_predictions")
    confidence_loss_mask = tf.placeholder(tf.float32, [None, NUMBER_CONFIDENCES], name="prediction_mask")
    true_locations = tf.placeholder(tf.float32, [None, NUMBER_LOCATIONS], name="true_locations")
    
    confidences = tf.reshape(confidences_all, [-1, NUMBER_CONFIDENCES, NUMBER_CLASSES])
    prediction_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=confidences, labels=true_confidences)
    prediction_loss *= confidence_loss_mask
    prediction_loss = tf.reduce_sum(prediction_loss)

    location_difference = true_locations - locations_all
    location_loss_l2 = .5 * (pow(location_difference, 2))
    location_loss_l1 = tf.abs(location_difference) - .5
    smooth_l1 = tf.less(tf.abs(location_difference), 1.0)
    location_loss = tf.where(smooth_l1, location_loss_l2, location_loss_l1)
    location_loss_mask = tf.minimum(true_confidences, 1)
    location_loss_mask = tf.to_float(location_loss_mask)

    location_loss_mask = tf.stack([location_loss_mask] * 4, axis=2)  
    location_loss_mask = tf.reshape(location_loss_mask, [-1, NUMBER_LOCATIONS])
    location_loss *= location_loss_mask
    location_loss = tf.reduce_sum(location_loss)

    loss = prediction_loss + location_loss + tf.reduce_sum(tf.losses.get_regularization_losses())
    
    tf.summary.histogram("location_difference", location_difference)
    tf.summary.histogram("logits", confidences)
    tf.summary.histogram("prediction_loss", prediction_loss)
    tf.summary.histogram("location loss", location_loss)
    tf.summary.histogram("loss", loss)

    all_probabilities = tf.nn.softmax(confidences)
    top_k_probabilities, top_k_prediction_probabilities = tf.nn.top_k(all_probabilities)
    top_k_probabilities = tf.reshape(top_k_probabilities, [-1, feature_map_number])
    top_k_prediction_probabilities = tf.reshape(top_k_prediction_probabilities, [-1, feature_map_number])
    
    return loss, true_confidences, true_locations, confidence_loss_mask, top_k_probabilities



def optimizer(loss):

    adam = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    return adam


