import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from model import ssd_layers, loss_function, load_vgg, optimizer 
from hyperparameters import *
import yaml
from data_input_output import create_boxes
import random
import scipy
import numpy as np
import warnings
from distutils.version import LooseVersion


assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def get_batch_function():
  
    images_list_dict = yaml.load(open(INPUT_YAML, 'rb').read())
    for i in range(len(images_list_dict)):
        images_list_dict[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(INPUT_YAML), images_list_dict[i]['path']))
    
    random.shuffle(images_list_dict)
    for batch_i in range(0, len(images_list_dict), BATCH_SIZE):
            
        Images, True_predictions, True_locations, Prediction_loss_masks = [], [], [], []
                    
        for i in range(len(images_list_dict[batch_i : batch_i+BATCH_SIZE])):

            image = scipy.misc.imread(images_list_dict[i]['path'])
            image = scipy.misc.imresize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
            print(image.shape)
            if image is None:
                raise IOError("Could not open", images_list_dict[i]['path']) 
            Images.append(image)

            true_prediction, true_location, prediction_loss_mask, default_box_matches_counter = create_boxes(images_list_dict[i])
            
            print("len true_prediction", len(true_prediction))
            print("len true_location", len(true_location))

            True_predictions.append(true_prediction)
            True_locations.append(true_location)
            Prediction_loss_masks.append(prediction_loss_mask)
        
       
        yield np.array(Images), np.array(True_predictions), np.array(True_locations), np.array(Prediction_loss_masks)


def run():

    with tf.Session() as sess:

        input_images, conv4_3, keep_prob = load_vgg(sess, VGG_PATH)
        predictions_all, predictions_locations_all = ssd_layers(input_images, conv4_3)
        loss_result, logits, true_predictions, true_locations, prediction_loss_mask = loss_function(predictions_all, predictions_locations_all)
        adam = optimizer(loss_result)

        sess.run(tf.global_variables_initializer())
        index = 0

        for i in range(EPOCHS):
            for images_generated, true_predictions_generated, true_locations_generated, prediction_loss_mask_generated in get_batch_function():

                # Forward pass
                _, loss = sess.run([adam, loss_result], feed_dict = {
                    input_images: images_generated,
                    true_predictions: true_predictions_generated, 
                    true_locations: true_locations_generated,
                    prediction_loss_mask: prediction_loss_mask_generated,
                    keep_prob: KEEP_PROB})

                if index % 5 == 0:
                    print("Epoch", i)
                    print("Loss {:.5f}...".format(loss))

                index += 1

         #TODO validation set
         #TODO timing / performance
         #TODO 

if __name__ == '__main__':

    run()