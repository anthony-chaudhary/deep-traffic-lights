
import tensorflow as tf
import tensorflow.contrib.slim as slim
import hyperparameters

def load_vgg(sess, vgg_path):

    tf.saved_model.loader.load(sess, 'vgg16', vgg_path)
    
    # TODO  confirm these names are right
    image = tf.get_default_graph().get_tensor_by_name('image_input:0')

    conv4_3 = tf.get_default_graph().get_tensor_by_name('conv4_3')

    return image, conv4_3


def prediction_and_location(net, layer_id, Predictions, Locations):

    with tf.variable_scope('prediction_and_location'+layer_id):
        prediction = slim.conv2d(net, hyperparameters.NUMBER_PREDICTIONS, [3,3], 
                                 activation_fn=None, scope='prediction')
        prediction = tf.contrib.layers.flatten(prediction)

        location = slim.conv2d(net, hyperparameters.NUMBER_LOCATIONS, [3,3], 
                               activation_fn=None, scope='location')
        location = tf.contrib.layers.flatten(location)

    Predictions.append(prediction)
    Locations.append(location)

    return Predictions, Locations


def ssd_layers(conv4_3, number_of_classes):

    Predictions, Locations = [], []

    with slim.arg_scope([slim.conv2d]):

        net = slim.conv2d(conv4_3, 1024, [3,3], scope='ssd_0')
        net = slim.conv2d(net, 1024, [1,1], scope='ssd_1')

        Predictions, Locations = prediction_and_location(net, 'ssd_1', Predictions, Locations)

        net = slim.conv2d(net, 256, [1,1], scope='ssd_2')
        net = slim.conv2d(net, 512, [3,3], 2, scope='ssd_2_s2')
        
        Predictions, Locations = prediction_and_location(net, 'ssd_2_s2', Predictions, Locations)

        net = slim.conv2d(net, 128, [1,1], scope='ssd_3')
        net = slim.conv2d(net, 256, [3,3], 2, scope='ssd_3_s2')
        
        Predictions, Locations = prediction_and_location(net, 'ssd_3_s2', Predictions, Locations)

    predictions_all = tf.concat(1, Predictions)
    locations_all = tf.concat(1, Locations)

    return predictions_all, locations_all