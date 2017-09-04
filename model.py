
import tensorflow as tf
import tensorflow.contrib.slim as slim
from hyperparameters import *

#from SSD-Tensorflow

def load_vgg(sess, vgg_path):

    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)

    images = tf.get_default_graph().get_tensor_by_name('image_input:0')
    conv4_3 = tf.get_default_graph().get_tensor_by_name('layer4_out:0')
    keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')

    return images, conv4_3, keep_prob


def prediction_and_location(net, layer_id, Predictions, Locations):

    with tf.variable_scope('prediction_and_location_'+layer_id):
        
        num_anchors = (NUMBER_DEFAULT_BOXES)

        prediction = slim.conv2d(net, num_anchors*NUMBER_CLASSES, [3,3], 
                                 activation_fn=None, scope='prediction', padding='SAME')
        prediction = tf.contrib.layers.flatten(prediction)

        location = slim.conv2d(net, num_anchors*4, [3,3], activation_fn=None, scope='location')
        location = tf.contrib.layers.flatten(location)

        Predictions.append(prediction)
        Locations.append(location)

        return Predictions, Locations


def ssd_layers(conv4_3):

    Predictions, Locations = [], []
    with tf.variable_scope("ssd_300"):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, 
                            weights_regularizer=slim.l2_regularizer(1e-2), padding='SAME'):

            # This seems to be causing some kind of bug so removed for now
            Predictions, Locations = prediction_and_location(conv4_3, 'ssd_vgg_0', Predictions, Locations)

            net = slim.conv2d(conv4_3, 1024, [3,3], scope='ssd_0')
            net = slim.conv2d(net, 1024, [1,1], scope='ssd_1')

            Predictions, Locations = prediction_and_location(net, 'ssd_1', Predictions, Locations)

            net = slim.conv2d(net, 256, [1,1], scope='ssd_2')
            net = slim.conv2d(net, 512, [3,3], 2, scope='ssd_2_s2')
        
            Predictions, Locations = prediction_and_location(net, 'ssd_2_s2', Predictions, Locations)

            net = slim.conv2d(net, 128, [1,1], scope='ssd_3')
            net = slim.conv2d(net, 256, [3,3], 2, scope='ssd_3_s2')

            Predictions, Locations = prediction_and_location(net, 'ssd_3_s2', Predictions, Locations)
        
        final_Predictions = tf.concat(Predictions, 1)
        final_Locations = tf.concat(Locations, 1)
        
        return final_Predictions, final_Locations


def loss_function(predictions_all, predictions_locations_all):

       
    true_predictions = tf.placeholder(tf.int32, [BATCH_SIZE, NUMBER_PREDICTIONS], name="true_predictions")
    true_locations = tf.placeholder(tf.float32, [BATCH_SIZE, NUMBER_LOCATIONS], name="true_locations")
    prediction_loss_mask = tf.placeholder(tf.float32, [BATCH_SIZE, NUMBER_PREDICTIONS], name="prediction_mask")

  
    c_pred_predictions = tf.reshape(predictions_all, [-1, NUMBER_PREDICTIONS, NUMBER_CLASSES])
    c_pred_locations = predictions_locations_all

    c_true_locations = true_locations
    c_true_predictions = true_predictions
    c_prediction_loss_mask = prediction_loss_mask

    tf.summary.histogram("logits", c_pred_predictions)

    prediction_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=c_pred_predictions,
                                                                    labels=c_true_predictions)
    prediction_loss *= c_prediction_loss_mask
    prediction_loss = tf.reduce_sum(prediction_loss)

    location_difference = c_true_locations - c_pred_locations

    location_loss_l2 = .5 * (pow(location_difference, 2))
    location_loss_l1 = tf.abs(location_difference) - .5
    smooth_l1 = tf.less(tf.abs(location_difference), 1.0)
    location_loss = tf.where(smooth_l1, location_loss_l2, location_loss_l1)
    
    
    location_loss_mask = tf.minimum(true_predictions, 1)
    location_loss_mask = tf.to_float(location_loss_mask)

    # or could multiple by classes / boxes to normalize?
    location_loss_mask = tf.stack([location_loss_mask] * 4, axis=2)  # Stacking locations for each prediction box?
    print(location_loss_mask)
    #location_loss_mask = tf.expand_dims(location_loss_mask, axis=-1)
    location_loss_mask = tf.reshape(location_loss_mask, [-1, NUMBER_LOCATIONS])
    
    print(location_loss_mask)
    print(location_loss)
    location_loss *= location_loss_mask
    

    location_loss = tf.reduce_sum(location_loss)

    loss = prediction_loss + location_loss + tf.reduce_sum(tf.losses.get_regularization_losses())
    
    all_probabilities = tf.nn.softmax(c_pred_predictions)
    top_k_probabilities, top_k_prediction_probabilities = tf.nn.top_k(all_probabilities)
    top_k_probabilities = tf.reshape(top_k_probabilities, [-1, feature_map_number])
    top_k_prediction_probabilities = tf.reshape(top_k_prediction_probabilities, [-1, feature_map_number])
    

    return loss, c_true_predictions, c_true_locations, prediction_loss_mask, top_k_probabilities



def optimizer(loss):

    adam = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    return adam


