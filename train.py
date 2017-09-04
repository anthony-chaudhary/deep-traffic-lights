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
    
    def get_batches_fn():
          # in hyper paramters
        #print(images_list_dict)

        for batch_i in range(0, len(images_list_dict), BATCH_SIZE):
            
            Images, True_predictions, True_locations, Prediction_loss_masks = [], [], [], []
                    
            for i in range(len(images_list_dict[batch_i : batch_i+BATCH_SIZE])):

                random.shuffle(images_list_dict)  ## rename this it's a list of dicts 
                image = scipy.misc.imread(images_list_dict[i]['path'])
                image = scipy.misc.imresize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
                #print(image.shape)
                if image is None:
                    raise IOError("Could not open", images_list_dict[i]['path'])
                image = image / 127.5 - 1. # normalize
                Images.append(image)

                true_prediction, true_location, prediction_loss_mask, default_box_matches_counter = create_boxes(images_list_dict[i])
            
                #TODO if default_box_matches_counter <= 0, get new images

                #print("true_prediction\t", true_prediction.shape)
                #print("true_location\t", true_location.shape)

                True_predictions.append(true_prediction)
                True_locations.append(true_location)
                Prediction_loss_masks.append(prediction_loss_mask)
        
                #True_predictions_ = np.concatenate(np.array(True_predictions))
                #Prediction_loss_masks_ = np.concatenate(np.array(Prediction_loss_masks))
                #True_locations_ = np.concatenate(np.array(True_locations))
            
            yield np.array(Images), True_predictions, True_locations, Prediction_loss_masks
    
    return get_batches_fn

def run():


    with tf.Session() as sess:

        
        input_images, conv4_3, keep_prob = load_vgg(sess, VGG_PATH)

        predictions_all, predictions_locations_all = ssd_layers(conv4_3)

        loss_result, true_predictions, true_locations, \
            prediction_loss_mask, top_k_probabilities = loss_function(predictions_all, predictions_locations_all)
        
        adam = optimizer(loss_result)

        get_batches_fn = get_batch_function()  # yields batches

        sess.run(tf.global_variables_initializer())
        index = 0

        train_writer = tf.summary.FileWriter('./tensorboard', sess.graph)
        saver = tf.train.Saver()

        for i in range(EPOCHS):
            for images_generated, true_predictions_generated, true_locations_generated, prediction_loss_mask_generated in get_batches_fn():

                # Forward pass

                #merge = tf.summary.merge_all()
                # summary, _, loss = sess.run([merge, adam, loss_result]

                _, loss = sess.run([adam, loss_result], feed_dict = {
                    input_images: images_generated,
                    true_predictions: true_predictions_generated, 
                    true_locations: true_locations_generated,
                    prediction_loss_mask: prediction_loss_mask_generated,
                    keep_prob: KEEP_PROB})

                #train_writer.add_summary(summary, index)

                print (index)

                if index % 5 == 0:
                    print("\n\nEpoch", i)
                    print("Loss {:.5f}...".format(loss))

                index += 1

        #TODO timing / performance
        saver.save(sess, "checkpoints/a.ckpt")
        print("Saved")


if __name__ == '__main__':

    run()