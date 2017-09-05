import os
import yaml

DEFAULT_BOXES = ((0.1, -0.3, 0.1, -.30), (.05, -0.15, .05, -0.15), (0.2, -.4, 0.2, .4), (0.1, -0.2, 0.1, -0.2))
FEATURE_MAP_SIZES = [[64, 64], [32, 32], [16, 16], [8, 8]]

NUMBER_DEFAULT_BOXES = len(DEFAULT_BOXES)
NUMBER_CLASSES = 3
NUMBER_CHANNELS = 3

#TRAINING
LEARNING_RATE = 1e-5
EPOCHS = 1
BATCH_SIZE = int(8)
KEEP_PROB = 1.
NEGATIVE_OVER_POSITIVE = 5

feature_map_number = 0
for f in FEATURE_MAP_SIZES:
    feature_map_number += (f[0] * f[1])

# Number of predictions does not include number of classes
# We assign each class a value [0, 1... n] 
NUMBER_PREDICTIONS = NUMBER_DEFAULT_BOXES * feature_map_number

NUMBER_LOCATIONS = 4 * NUMBER_DEFAULT_BOXES * feature_map_number

print("feature_map_number\t", feature_map_number)
print("NUMBER_PREDICTIONS\t", NUMBER_PREDICTIONS)
print("NUMBER_LOCATIONS\t", NUMBER_LOCATIONS)


#Data prep
IOU_THRESHOLD = .4  # Goal to be .5 as per paper
CONFIDENCE_THRESHOLD = .5
TEST_IOU_THRESHOLD = .1

INPUT_YAML = "data/dataset_train_rgb/train.yaml"

data_dir = './data'

VGG_PATH = os.path.join(data_dir, 'vgg')
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

images_list_dict = yaml.load(open(INPUT_YAML, 'rb').read())
for i in range(len(images_list_dict)):
    images_list_dict[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(INPUT_YAML), images_list_dict[i]['path']))
 
images_list_dict = images_list_dict[ :1600]


# Background_class_is : 0 
LABEL_DICT =  {
    "Green" : 1,
    "Red" : 2,
    "GreenLeft" : 1,
    "GreenRight" : 1,
    "RedLeft" : 2,
    "RedRight" : 2,
    "Yellow" : 0,
    "off" : 0,
    "RedStraight" : 2,
    "GreenStraight" : 1,
    "GreenStraightLeft" : 1,
    "GreenStraightRight" : 1,
    "RedStraightLeft" : 2,
    "RedStraightRight" : 2
    }
