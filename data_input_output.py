import numpy as np
from hyperparameters import *


def calc_iou(box_a, box_b):
	"""
    FROM https://github.com/georgesung/ssd_tensorflow_traffic_sign_detection

	Calculate the Intersection Over Union of two boxes
	Each box specified by upper left corner and lower right corner:
	(x1, y1, x2, y2), where 1 denotes upper left corner, 2 denotes lower right corner

	Returns IOU value
	"""
	# Calculate intersection, i.e. area of overlap between the 2 boxes (could be 0)
	# http://math.stackexchange.com/a/99576
	x_overlap = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
	y_overlap = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
	intersection = x_overlap * y_overlap

	# Calculate union
	area_box_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
	area_box_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
	union = area_box_a + area_box_b - intersection

	iou = intersection / union
	return iou



def create_prediction_loss_mask(true_prediction):

    number_positive = np.where(true_prediction > 0)[0].shape[0]
    number_negative = NEGATIVE_OVER_POSITIVE * number_positive
    true_prediction_size = np.sum(true_prediction.shape)
    print(number_negative, number_negative, true_prediction_size)
    print(true_prediction.shape)

    if number_positive + number_negative < true_prediction_size:

        prediction_loss_mask = np.copy(true_prediction)
        prediction_loss_mask[np.where(prediction_loss_mask > 0)] = 1.

        zero_indices = np.where(prediction_loss_mask == 0.)
        zero_indices = np.transpose(zero_indices)

        choosen_zero_indices = zero_indices[np.random.choice(zero_indices.shape[0], int(number_negative), False)]

        print(prediction_loss_mask.shape)

        for z in choosen_zero_indices:
            i, j = z
            prediction_loss_mask[i][j] = 1.

    else:
        prediction_loss_mask = np.ones_like(true_prediction)

    return prediction_loss_mask


def create_boxes(image_dict):

    #if image_dict['boxes'] != None:

    classes = []
    relative_coordinates = []

    for box in image_dict['boxes']:
        
        integer_label = LABEL_DICT[box['label']]  # Convert string labels to integers
        classes.append(integer_label)
        print(box['label'])
        coordinates = np.array([box['x_min'], box['y_min'], box['x_max'], box['y_max']])
        scale = np.array([1280, 720, 1280, 720])
        relative_coordinates.append(coordinates / scale)

    print(len(relative_coordinates))
    # Init
    true_length = 0
    for f in FEATURE_MAP_SIZES:
        true_length += f[0] * f[1] * NUMBER_DEFAULT_BOXES

    true_prediction = np.zeros(true_length)
    true_location = np.zeros(true_length * 4)

    default_box_matches_counter = 0

    for i, gt_coordinates in enumerate(relative_coordinates):
        prediction_index = 0
        for f in FEATURE_MAP_SIZES:
            print("prediction_index", prediction_index)
            print("default_box_matches_counter", default_box_matches_counter)
            
            for row in range(f[0]):
                for col in range(f[1]):
                    for d in DEFAULT_BOXES:
                        
                        x1_offset, y1_offset, x2_offset, y2_offset = d
                        #print(d, "\n")

                        a = np.array([
                            max(0, col + x1_offset),
                            max(0, row + y1_offset),
							min(f[1], col+1 + x2_offset),
							min(f[0], row+1 + y2_offset) ])
                        scale = np.array([f[1], f[0], f[1], f[0]])
                        default_coordinates = a / scale
                        #print(default_coordinates, gt_coordinates)

                        iou = calc_iou(gt_coordinates, default_coordinates)
                        #print(iou)

                        if iou >= IOU_THRESHOLD:

                            print(f, d)
                            print(default_coordinates, gt_coordinates)
                            print(iou)

                            true_prediction[prediction_index] = classes[i]
                            
                            default_box_matches_counter += 1

                            center = np.array([col + .5, row + .5])
                            absolute_gt_coordinates = gt_coordinates * scale
                            new_coordinates = absolute_gt_coordinates - np.concatenate((center, center))
                            true_location[prediction_index*4 : prediction_index*4 + 4] = new_coordinates

                        prediction_index += 1
    
    prediction_loss_mask = create_prediction_loss_mask(true_prediction)

    return true_prediction, true_location, prediction_loss_mask, default_box_matches_counter

