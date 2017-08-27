import os


DEFAULT_BOXES = ((-0.5, -0.8, 0.4, 0.5), (0.1, -0.1, -0.1, 0.1), (-0.05, -0.1, 0.05, 0.05), (-0.2, -0.8, 0.2, 0.2))
FEATURE_MAP_SIZES = [[38, 38], [19, 19], [10, 10], [5, 5]]
# 38 for VGG, then rest for SSD layers

NUMBER_DEFAULT_BOXES = len(DEFAULT_BOXES)
NUMBER_CLASSES = 3
NUMBER_CHANNELS = 3

feature_map_number = 0
for f in FEATURE_MAP_SIZES:
    feature_map_number += (f[0] * f[1])

NUMBER_PREDICTIONS_GROUND_TRUTH = NUMBER_DEFAULT_BOXES * feature_map_number * NUMBER_CHANNELS
NUMBER_PREDICTIONS = NUMBER_CLASSES * NUMBER_DEFAULT_BOXES * feature_map_number * NUMBER_CHANNELS
NUMBER_PREDICTIONS_PER_FEATURE_MAP = 1

NUMBER_LOCATIONS = (NUMBER_CLASSES + 4) * NUMBER_DEFAULT_BOXES * feature_map_number * NUMBER_CHANNELS
NUMBER_LOCATIONS_PER_FEATURE_MAP = 1

TOTAL_OUPUTS = (NUMBER_CLASSES + 4) * NUMBER_DEFAULT_BOXES * feature_map_number * NUMBER_CHANNELS

print("feature_map_number\t", feature_map_number)
print("NUMBER_PREDICTIONS\t", NUMBER_PREDICTIONS)
print("NUMBER_LOCATIONS\t", NUMBER_LOCATIONS)
print("NUMBER_PREDICTIONS_PER_FEATURE_MAP\t", NUMBER_PREDICTIONS_PER_FEATURE_MAP)
print("NUMBER_LOCATIONS_PER_FEATURE_MAP\t", NUMBER_LOCATIONS_PER_FEATURE_MAP)

#TRAINING
LEARNING_RATE = 1e-4
EPOCHS = 1
BATCH_SIZE = 1
KEEP_PROB = 1.
NEGATIVE_OVER_POSITIVE = 3

#Data prep
IOU_THRESHOLD = .1  # Goal to be .5 as per paper

INPUT_YAML = "data/dataset_train_rgb/train.yaml"

data_dir = './data'
VGG_PATH = os.path.join(data_dir, 'vgg')
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300


# Background_class_is : 0 
LABEL_DICT =  {
    "Green" : 1,
    "Red" : 2,
    "GreenLeft" : 3,
    "GreenRight" : 4,
    "RedLeft" : 5,
    "RedRight" : 6,
    "Yellow" : 7,
    "off" : 8
    }
