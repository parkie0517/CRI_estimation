"""
This file is used to filter semantic segmentation results.

"""


colors = { 
    'null': [  0,   0,   0], # null
    'road': [128, 64, 128], #road
    'sidewalk': [244, 35, 232], #sidewalk
    'building':[70, 70, 70], #building
    'wall':[102, 102, 156],#wall
    'fence':[190, 153, 153],#fence
    'pole':[153, 153, 153],#pole
    'traffic_light':[250, 170, 30],#traffic light
    'traffiic_sign':[220, 220, 0],#traffiic sign
    'vegetation':[107, 142, 35],  # vegetation dark green
    'terrain':[152, 251, 152],  # terrain bright green
    'sky':[0, 130, 180],#sky
    'person':[220, 20, 60], #person
    'rider':[255, 0, 0], # rider
    'car':[0, 0, 142],
    'truck':[0, 0, 70],
    'bus':[0, 60, 100],
    'train':[0, 80, 100],
    'motorcycle':[0, 0, 230], # motorcycle
    'bicycle':[119, 11, 32], # bicycle
}

# load images from the directory
src_path = '/home/vilab/ssd1tb/hj_ME455/Term_Project/results/segmentation/color'
dst_path = '/home/vilab/ssd1tb/hj_ME455/Term_Project/results/segmentation/filtered'
image_list = ???
for image_path in image_list:
    # open image
    
    # load image
     
    # fill a rectangle with [0, 0, 0]
        # left top (448, 1024-50)
        # left bottom (448, 1024)
        # right top (2048-448, 1024-50)
        # right bottom (2048-224, 1024)
    
    
# if pixel is not within the argument then change to black
