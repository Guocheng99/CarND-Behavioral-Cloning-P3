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
                moreImgs, moreAngles = augment_data(img,angle=angle)
                images = images + moreImgs
                angles = angles + moreAngles

            inputs = np.array(images)
            outputs = np.array(angles)

            yield shuffle(inputs,outputs)


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


def newModel11():
    '''
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
    '''

    model = pre_process_layers()
    model.add(Convolution2D(3, 1, 1,border_mode='valid', init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(32, 3, 3,border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(32, 3, 3,border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3,border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3,border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3,border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(128, 3, 3,border_mode='valid', init='he_normal'))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(512, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(64, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(16, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(1, init='he_normal'))

    return model


def train_model(model, features, labels, epochs = 3):
    model.compile(loss='mse', optimizer='adam')
    model.fit(features, labels, validation_split=0.2, shuffle=True, nb_epoch=epochs)

def train_model_with_generator(model, train_dataset, valid_dataset,batch=128,epochs=3):
    train_generator = feed_data_generator(dataset=train_dataset,batch_size=batch)
    valid_generator = feed_data_generator(dataset=valid_dataset,batch_size=batch)

    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(train_generator,samples_per_epoch=len(train_dataset)*5,
                                  nb_epoch=epochs,validation_data=valid_generator,
                                  nb_val_samples=len(valid_dataset),verbose=1)
    return history

def save_model(model, modelfile):
    model.save(modelfile)
    print("Model saved at " + modelfile)

# model = LeNet5()
#model = newModel11()
model = nVidia9()


print('Loading images')
train_dataset, valid_dataset = find_all_dataset('./data', correction=0.25)

print('Training model')
history = train_model_with_generator(model,train_dataset=train_dataset,
                                     valid_dataset=valid_dataset,
                                     batch=32,epochs=10)
save_model(model,'run6.h5')
print()

# print(history.history.keys())
print('Loss')
print(history.history['loss'])
print('Validation Loss')
print(history.history['val_loss'])
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