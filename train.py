import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from model import ssd_layers, loss, load_vgg, optimizer 
from hyperparameters import *
import yaml
from data_input_output import create_boxes

def get_batch_function():
  
    input_yaml = "data/dataset_train_rgb/train.yaml"
    images_list_dict = yaml.load(open(input_yaml, 'rb').read())
    for i in range(len(image_dict)):
        images_list_dict[i] = os.path.abspath(os.path.join(os.path.dirname(input_yaml), images_list_dict[i]['path']))
    
    random.shuffle(images_list_dict)
    for batch_i in range(0, len(images_list_dict), batch_size):
            
        Images, True_predictions, True_locations, Prediction_loss_masks = [], [], [], []
                    
        for i in range(len(images_list_dict[batch_i:batch_i+batch_size])):

            Images = scipy.misc.imread(images_list_dict[i])
            images.append(image)

            true_prediction, true_location, prediction_loss_mask, default_box_matches_counter = create_boxes(images_list_dict[i])
            True_predictions.append(true_prediction)
            True_locations.append(true_location)
            Prediction_loss_masks.append(prediction_loss_mask)

        yield np.array(Images), np.array(True_predictions), np.array(True_locations), np.array(Prediction_loss_masks)


def run():

    with tf.Graph().as_default() and tf.Session() as sess:

        image, conv4_3 = load_vgg(sess, VGG_PATH)
        predictions_all, predictions_locations_all = ssd_layers(image, conv4_3)
        loss, logits, true_prediction, true_locations = loss(predictions_all, predictions_locations_all)
        adam = optimizer(loss)

        sess.run(tf.global_variables_initializer())

        for i in range(EPOCHS):
            for images_generated, true_predictions_generated, true_locations_generated, prediction_loss_mask_generated in get_batch_function():

                # Forward pass
                _, loss = sess.run([adam, loss], feed_dict = {
                    images: images_generated,
                    true_prediction: true_predictions_generated, 
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

if __name__ == '__train__':

    run()