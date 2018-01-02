from funcs import *
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Lambda,Convolution2D,Flatten,Dense,Cropping2D,Dropout,ELU
from keras.layers.pooling import MaxPooling2D

import matplotlib.pyplot as plt


def feed_data_generator(dataset,batch_size=128):
    num_data = len(dataset)
    while 1:
        dataset = shuffle(dataset)
        for offset in range(0,num_data,batch_size):
            batch = dataset[offset:offset+batch_size]
            images = []
            angles = []
            for imgPath,angle in batch:
                img = cv2.imread(imgPath)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                images.append(img)
                angles.append(angle)
                moreImgs, moreAngles = augment_data_single(img,angle=angle)
                images = images + moreImgs
                angles = angles + moreAngles

            inputs = np.array(images)
            outputs = np.array(angles)

            yield shuffle(inputs,outputs)

def feed_data_generator_single_line(dataset,batch_size=128):
    num_data = len(dataset)
    while 1:
        dataset = shuffle(dataset)
        for offset in range(0,num_data,batch_size):
            batch = dataset[offset:offset+batch_size]
            images = []
            angles = []
            for imgPath_c, imgPath_l, imgPath_r,angle in batch:
                new_img, new_angle = pick_one_to_transform_for_generator(imgPath_c, imgPath_l, imgPath_r, angle)
                images.append(new_img)
                angles.append(new_angle)

            inputs = np.array(images)
            outputs = np.array(angles)
            yield shuffle(inputs,outputs)

def pick_one_to_transform_for_generator(img_c, img_l, img_r, angle):
    i_lrc = np.random.randint(3)
    if (i_lrc == 0):
        path_file = img_l
        shift_ang = .25
    if (i_lrc == 1):
        path_file = img_c
        shift_ang = 0.
    if (i_lrc == 2):
        path_file = img_r
        shift_ang = -.25

    new_img = cv2.imread(path_file)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    new_angle = angle+shift_ang

    new_img, new_angle = transform_data_single(new_img, angle=new_angle)

    return new_img,new_angle


def pre_process_layers():
    model = Sequential()
    model.add(Lambda(lambda x:(x/255.0)-0.5,input_shape=(160,320,3)))
    model.add(Cropping2D(cropping= ((30,25),(0,0))))
    return model

def LeNet5():
    model = pre_process_layers()
    model.add(Convolution2D(6,5,5,activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(6,5,5,activation='relu',border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def nVidia9():
    '''
    https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    '''
    model = pre_process_layers()
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu', init='he_normal'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu', init='he_normal'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu', init='he_normal'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64,3,3,activation='relu', init='he_normal'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64,3,3,activation='relu', init='he_normal'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(100, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Dense(50, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Dense(10, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))

    model.add(Dense(1, init='he_normal'))

    return model

def newModel14():
    '''
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    '''

    model = pre_process_layers()
    model.add(Convolution2D(3, 1, 1,border_mode='valid',name='conv0', init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(32, 3, 3,border_mode='valid',name='conv1', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(32, 3, 3,border_mode='valid',name='conv2', init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3,border_mode='valid',name='conv3', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3,border_mode='valid',name='conv4', init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3,border_mode='valid',name='conv5', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(128, 3, 3,border_mode='valid',name='conv6', init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(512, name='hidden1', init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(64, name='hidden2', init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(16, name='hidden3', init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(1, name='output', init='he_normal'))

    return model


def train_model(model, features, labels, epochs = 3):
    model.compile(loss='mse', optimizer='adam')
    model.fit(features, labels, validation_split=0.2, shuffle=True, nb_epoch=epochs)

def train_model_with_generator(model, train_dataset, valid_dataset,batch=128,epochs=3):
    train_generator = feed_data_generator(dataset=train_dataset,batch_size=batch)
    valid_generator = feed_data_generator(dataset=valid_dataset,batch_size=batch)

    # train_generator = feed_data_generator_single_line(dataset=train_dataset,batch_size=batch)
    # valid_generator = feed_data_generator_single_line(dataset=valid_dataset,batch_size=batch)

    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(train_generator,samples_per_epoch=len(train_dataset)*2,
                                  nb_epoch=epochs,validation_data=valid_generator,
                                  nb_val_samples=len(valid_dataset),verbose=1)
    return history

def save_model(model, modelfile):
    model.save(modelfile)
    print("Model saved at " + modelfile)

print('Loading images')
train_dataset, valid_dataset = find_all_dataset('./data', correction=0.25)
#train_dataset, valid_dataset = find_all_dataset_single_line('./data')

# model = LeNet5()
model = nVidia9()
#model = newModel14()

# from keras.utils.visualize_util import plot
print('Model Summary')
print(model.summary())
# plot(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
print()

# X_tmp = feed_data_generator(train_dataset,batch_size=32)

print('Training model')
history = train_model_with_generator(model,train_dataset=train_dataset,
                                     valid_dataset=valid_dataset,
                                     batch=32,epochs=10)
save_model(model,'run6.h5')
print()

print(history.history.keys())
print('Loss')
print(history.history['loss'])
print('Validation Loss')
print(history.history['val_loss'])
print()

print('The End')
print()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.grid(b='on')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()