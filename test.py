from hyperparameters import *
from data_input_output import calc_iou
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import scipy
import tensorflow as tf
from model import ssd_layers, loss_function, load_vgg, optimizer
import time
import numpy as np
import cv2

def get_test_output():
    
    # handle multiple?
    pass


def run():

    with tf.Session() as sess:
        
        # refactor to share with train
        input_images, conv4_3, conv5_3, keep_prob = load_vgg(sess, VGG_PATH)
        predictions_all, predictions_locations_all = ssd_layers(conv4_3, conv5_3)
        loss_result, true_predictions, true_locations, \
            prediction_loss_mask, top_k_probabilities = loss_function(predictions_all, predictions_locations_all)
        adam = optimizer(loss_result)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        print("Model restored")
        
        run_image(sess, input_images, predictions_all, predictions_locations_all, top_k_probabilities)
        

def run_image(sess, input_images, predictions_all, predictions_locations_all, top_k_probabilities):

    """
    image_file_path, string

    """
    
    image_file_path = "22360.png"
    image = scipy.misc.imread(image_file_path)
    if image is None:
        raise IOError("Could not open", image_file_path)

    image_modified = scipy.misc.imresize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
    image_modified = image_modified / 127.5 - 1. # normalize
    image_modified = [image_modified, image_modified]  # Change to be batch size of 1!
    image_modified = np.array(image_modified)  # batch of 1

    t0 = time.time()

    confidence_out, locations_out, probabilities_out = sess.run([predictions_all, 
                                                                 predictions_locations_all, 
                                                                 top_k_probabilities],
                                                                feed_dict={input_images: image_modified})

    boxes = [locations_out[0], confidence_out[0].astype("float32"), probabilities_out[0]]

    print(boxes)

    # TODO add NMS here.

    t1 = time.time()

    print("Inference time (seconds)", t1 - t0)

    scale = np.array([1280/IMAGE_WIDTH, 720/IMAGE_HEIGHT, 1280/IMAGE_WIDTH, 720/IMAGE_HEIGHT])
    
    #if len(boxes) > 0:
        #boxes[ :, : 4] = boxes[:, :4] * scale

    # Drawing
    for b in boxes:

        b_coords = [round(x) for x in b[:4]]
        b_coords = b_coords*scale
        b_coords = [int(x) for x in b_coords]  # refactor this bad way to do it
        print(b_coords)
        cls = int(b[4])
        cls_prob = b[5]

        image = cv2.rectangle(image, tuple(b_coords[:2]), tuple(b_coords[2:]), (0,255,0))
        label_str = '%s %.2f' % (cls, cls_prob)
        image = cv2.putText(image, label_str, (b_coords[0], b_coords[1]), 0, 0.5, (0,255,0), 1, cv2.LINE_AA)
    
    save_samples(image)

    return 0


def save_samples(image):
    
    out = os.path.join("runs", str(time.time()))
    os.makedirs(out)
    print("Saving samples to", out)

    # can change to save multiple here if wanted
    scipy.misc.imsave(os.path.join(out, "22360.png"), image)



def nms(confidences, locations, top_k_probabilities):
    """
        confidences,  array of class confidences for a prediction
        locations, array of locations (x, y offsets)
        prob

    """
    i = 0

    for f in FEATURE_MAP_SIZES:
        h, w = f
        for row in range(h):
            for col in range(w):
                for d in DEFAULT_BOXES:
                    
                    print(top_k_probabilities[i])

                    if confidences[i] > 0 and top_k_probabilities[i] > CONFIDENCE_THRESHOLD:

                        x_c, y_c = col + .5, row +.5
                        center = np.array(x_c, y_c, x_c, y_c)
                        abs_coords = center + confidences[i : i +4]

                        w_scale = IMAGE_WIDTH * w
                        h_scale = IMAGE_HEIGHT * h
                        scale = np.array(w_scale, h_scale, w_scale, h_scale)
                        box_coordinates = abs_coords * scale
                        box_coordinates_array = [int(round(x)) for x in box_coordinates]

                        # Compare

                        c = confidences[i]  # Class
                        class_probability = top_k_probabilities[i]
                        box = (*box_coordinates_array, classes, class_probability)

                        if (len(class_boxes) == 0):  # Init case
                            class_boxes[c].append(box)
                        else:
                            suppresed = False
                            overlap = False

                            for j in class_boxes[c]:

                                iou = calc_iou(box[ : 4], j[:4])

                                if iou > TEST_IOU_THRESHOLD:

                                    overlap = True

                                    if box[5] > j[5]: #confidence
                                        class_boxes[c].remove(j)
                                        suppresed = True
                            if suppresed or not overlap:
                                class_boxes[c].append(box)
                    i += 1
    boxes = []
    for c in class_boxes:
        for c_box in class_boxes[c]:
            boxes.append(c_box)
    boxes = np.array(boxes)

    return boxes


if __name__ == '__main__':

    run()