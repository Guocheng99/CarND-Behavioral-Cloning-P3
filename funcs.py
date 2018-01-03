import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from pandas.io.parsers import read_csv


def find_all_dataset(basePath,correction=0.25):
    lines = read_csv(basePath + '/driving_log.csv').values
    images = []
    angles = []
    for line in lines:
        angle = float(line[3])
        # Center image
        images.append(basePath + '/' + line[0].strip())
        angles.append(angle)
        # Left image
        images.append(basePath + '/' + line[1].strip())
        angles.append(angle + correction)
        # Right image
        images.append(basePath + '/' + line[2].strip())
        angles.append(angle - correction)

    dataset = list(zip(images,angles))
    print(len(dataset))

    train_dataset, valid_dataset = train_test_split(dataset,test_size=0.2)

    return train_dataset,valid_dataset

def new_random_brightness_image(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def shift_image(image, angle, move=150):
    # Translation
    rx = move * np.random.uniform(low=-0.5,high=0.5)
    ry = 10 * np.random.uniform(low=-0.5,high=0.5)
    M = np.float32([[1, 0, rx], [0, 1, ry]])

    image1 = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    angle1 = angle + rx / move * 2 * 0.2

    return image1, angle1

def add_random_shadow_image(image):
    '''
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    '''
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = (shadow_mask == 1)
        cond0 = (shadow_mask == 0)
        if np.random.randint(2) == 1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image1 = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image1

def augment_data(image,angle):
    '''
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    '''
    images = []
    angles = []

    # Flipping
    images.append(cv2.flip(image,1))
    angles.append(angle * -1.0)

    # Brightness
    images.append(new_random_brightness_image(image))
    angles.append(angle)

    # Translation for driving on slop
    image1, angle1 = shift_image(image, angle, move=150)
    images.append(image1)
    angles.append(angle1)

    # shadow
    image2 = add_random_shadow_image(image)
    images.append(image2)
    angles.append(angle)

    return images,angles

def augment_data_single(image,angle):
    '''
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    '''

    images = []
    angles = []

    # Brightness
    img = new_random_brightness_image(image)

    # Translation for driving on slop
    img, angle1 = shift_image(img, angle, move=150)

    # shadow
    img = add_random_shadow_image(img)

    # Flipping
    flip = np.random.randint(2)
    if flip == 0:
        img = cv2.flip(img, 1)
        angle1 = angle1 * -1.0

    images.append(img)
    angles.append(angle1)

    return images, angles
