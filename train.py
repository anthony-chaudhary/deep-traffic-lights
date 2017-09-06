import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from model import ssd_layers, loss_function, load_vgg, optimizer 
from hyperparameters import *
import yaml
from data_input_output import create_boxes, get_batch_function
import warnings
from distutils.version import LooseVersion

assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def run():

    with tf.Session() as sess:   
        
        input_images, conv4_3_pool, conv4_3_relu, keep_prob = load_vgg(sess, VGG_PATH)
        confidences_all, locations_all = ssd_layers(conv4_3_pool, conv4_3_relu)
        loss_result, true_predictions, true_locations, \
            prediction_loss_mask, top_k_probabilities = loss_function(confidences_all, locations_all)
        
        adam = optimizer(loss_result)
        get_batches_fn = get_batch_function()  # yields batches

        sess.run(tf.global_variables_initializer())
        index = 0

        train_writer = tf.summary.FileWriter('./tensorboard', sess.graph)
        saver = tf.train.Saver()

        for i in range(EPOCHS):
            for images_generated, true_predictions_generated, true_locations_generated, prediction_loss_mask_generated in get_batches_fn():

                # Forward pass

                merge = tf.summary.merge_all()
                
                summary, _, loss = sess.run([merge, adam, loss_result], feed_dict = {
                    input_images: images_generated,
                    true_predictions: true_predictions_generated, 
                    true_locations: true_locations_generated,
                    prediction_loss_mask: prediction_loss_mask_generated,
                    keep_prob: KEEP_PROB})

                train_writer.add_summary(summary, index)

                print(index)

                if index % 5 == 0:
                    print("\n\nEpoch", i)
                    print("Loss \t {:.5f}...\n\n".format(loss))

                index += 1

        #TODO timing / performance
        #TODO add to change to time
        saver.save(sess, "checkpoints/a.ckpt")
        print("Saved")


if __name__ == '__main__':

    run()