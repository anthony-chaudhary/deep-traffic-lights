from hyperparameters import *
from data_input_output import calc_iou

def get_test_output():
    pass
    #TODO



def save_samples():
    pass


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