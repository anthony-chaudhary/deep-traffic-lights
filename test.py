from hyperparameters import *

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

                        class = confidences[i]
                        class_probability = top_k_probabilities[i]

