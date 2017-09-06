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
        loss, probabilities, probability_confidences , \
            true_locations, true_confidences, confidence_loss_mask = loss_function(confidences_all, locations_all)
        
        adam = optimizer(loss)
        get_batches_fn = get_batch_function()  # yields batches

        sess.run(tf.global_variables_initializer())
        index = 0

        train_writer = tf.summary.FileWriter('./tensorboard', sess.graph)
        saver = tf.train.Saver()

        for i in range(EPOCHS):
            for images_generated, true_confidences_generated, true_locations_generated, confidence_loss_mask_generated in get_batches_fn():

                # Forward pass

                merge = tf.summary.merge_all()
                
                summary, _, loss_out = sess.run([merge, adam, loss], feed_dict = {
                    input_images: images_generated,
                    true_confidences: true_confidences_generated, 
                    true_locations: true_locations_generated,
                    confidence_loss_mask: confidence_loss_mask_generated,
                    keep_prob: KEEP_PROB})

                train_writer.add_summary(summary, index)

                if index % 50 == 0:
                    print("Saved")
                
                    print("\n\nEpoch", i, "index", index)
                print("Loss \t {:.5f}...\n\n".format(loss_out))

                index += 1
            
            # probably want a better way to do that...
            # and maybe change tensorboard thing to big drive
            # saver.save(sess, "checkpoints/c.ckpt")
            # print("Saved")
        # TODO timing / performance
        # TODO add to change to time
        # saver.save(sess, "checkpoints/b.ckpt")
        #print("Saved")


if __name__ == '__main__':

    run()