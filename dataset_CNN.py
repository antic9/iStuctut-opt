import csv
import sys
from tabnanny import verbose

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

def set_acc(struc):
    convs = struc[0]
    convs = convs.replace('conv','')
    # convs = convs.replace(',','')

    strides = struc[1]
    strides = strides.replace('stride','')

    paddings = struc[2]
    paddings = paddings.replace('padding','')
    # paddings = paddings.replace(';','')

    # # print(struc)

    pooling = struc[3]
    pooling = pooling.replace('_pooling22','')
    # print(pooling)

    pool_strides=struc[4]
    pool_strides = pool_strides.replace('stride','')

    afunc = struc[5]
    afunc = afunc.lower()
    # exit(0)
    # convs = 5
    # strides = 2
    accuracy = []
    loss=[]
    conv=int(int(convs)/11)
    # print(type(conv))
    stride=int(strides)
    padding=int(paddings)
    pool_stride = int(pool_strides)
    activation=afunc
    # activation='relu'
    # padding_input=(padding,padding)
    train_imageswpadding=layers.ZeroPadding2D(padding=padding)(train_images)
    test_imageswpadding=layers.ZeroPadding2D(padding=padding)(test_images)
    # print(test_imageswpadding.shape())
    # print(conv)
    # print(strides)
    # print(paddings)
    # print(pool_strides)
    # print(afunc)
    model = models.Sequential()

    model.add(layers.Conv2D(32, (conv, conv),strides = stride,  input_shape=(32+padding*2, 32+padding*2, 3)))

    if pooling == "average":
        model.add(layers.AveragePooling2D(pool_size=(2, 2),strides=(pool_stride,pool_stride)))
    elif pooling == "max":
        model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=(pool_stride,pool_stride)))
    else:
        exit(0)
    # print("a")

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation=activation))
    model.add(layers.Dense(10))

    # model.summary()

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    history = model.fit(train_imageswpadding, train_labels, epochs=10, verbose = 0, validation_data=(test_imageswpadding, test_labels))

    test_loss, test_acc = model.evaluate(test_imageswpadding,  test_labels, verbose=2)

    return test_acc
    # print("test accuracy : {}".format(test_acc))
    # print("test loss : {}".format(test_loss))
    # accuracy.append(test_acc)
    # loss.append(test_loss)
    # print(conv)

    # with open('acc.csv', 'a',newline='') as f:
    #     for i in range(len(accuracy)):
    #         w = csv.writer(f)
    #         print(accuracy[i])
    #         w = w.writerow(accuracy[i])
    # with open('loss.csv', 'a',newline='') as f:
    #     w = csv.writer(f)
    #     for i in range(len(accuracy)):
    #         w = w.writerow(loss[i])


    # print(accuracy)
    # print("--------------------------")
    # print(loss)
    # exit(0)
    return 0

