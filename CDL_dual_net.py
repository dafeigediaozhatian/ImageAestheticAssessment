#!/usr/bin/python
#-*- coding:utf-8 -*-
import h5py
import numpy as np
import os
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
"""
定义网络结构
"""
def Comparative_Deep_Learning_of_Hybrid_Representations_network():
    """
     定义图片处理的CNN模块
     输入 256*256*3
     输出 1024
    """
    digit_input = Input(shape=(256, 256, 3))
    x = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='same', activation='relu',name ='con1')(digit_input)
    x = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same', data_format=None,name ='pool1')(x)
    x = Conv2D(filters=96, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu',name ='con2')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=None,name ='pool2')(x)
    x = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',name ='con3')(x)
    x = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',name ='con4')(x)
    x = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',name ='con5')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=None,name ='pool3')(x)
    x = keras.layers.Flatten()(x)
    x = Dense(4096, activation='relu',name ='dense1')(x)
    x = Dense(4096, activation='relu',name ='dense2')(x)
    out = Dense(1024, activation='relu',name ='dense3')(x)
    vision_model = Model(inputs=digit_input, outputs=out,name = "image_mode")
    """
     定义用户模块
     输入 1024
     输出 1024
    """
    user_input = Input(shape=(1024,))
    x = Dense(1024, activation='relu',name ='dense4')(user_input)
    x = Dense(2048, activation='relu',name ='dense5')(x)
    x = Dense(4096, activation='relu',name ='dense6')(x)
    user_out = Dense(1024, activation='relu',name ='dense7')(x)
    user_model = Model(inputs=user_input, outputs=user_out,name ="user_mode")

    """
     定义整体网络结构
    """
    input_good_image = Input(shape=(256, 256, 3))
    input_bed_image = Input(shape=(256, 256, 3))
    input_User_vec = Input(shape=(1024,))
    # 视觉模块完全共享，包括参数
    good_image_tensor = vision_model(input_good_image)
    bed_image_tensor = vision_model(input_bed_image)
    User_tensor = user_model(input_User_vec)
    # print good_image_tensor._keras_shape
    # print bed_image_tensor._keras_shape
    # print User_tensor._keras_shape
    d1 = keras.layers.subtract([good_image_tensor, User_tensor],)
    d2 = keras.layers.subtract([bed_image_tensor, User_tensor])
    d1 = keras.layers.multiply([d1, d1],)
    d2 = keras.layers.multiply([d2, d2])
    x1 = Dense(1, activation='sigmoid',name ='dense8')(d1)
    x2 = Dense(1, activation='sigmoid',name ='dense9')(d2)
    out = keras.layers.subtract([x2, x1])

    model = Model(inputs=[input_good_image, input_bed_image, input_User_vec], outputs=out)
    return model
"""
定义训练数据生成器
"""
def generate_trian_sequences(train_triple_file, user_Data_file, img_Data_file, per_epoch_steps, batche_size =64, shuffle=False):
    batche = 0
    while True:
        if(batche > per_epoch_steps):
            batche = 0
        train_triple = train_triple_file['train_triple_let'][batche_size * batche : batche_size * (batche + 1),: ].astype(np.int32)
        # 原始id数组
        # User_id_array = train_triple[:, 0]
        # good_image_id_array = train_triple[:, 1]
        # bed_image_id_array= train_triple[:, 2]
        # # 原始id列表
        # User_id_list = list(User_id_array)
        # good_image_id_list = list(good_image_id_array)
        # bed_image_id_list=list(bed_image_id_array)
        # id集合可还原去重
        User_id_set = list(set(train_triple[:, 0]))
        good_image_id_set = list(set(train_triple[:, 1]))
        bed_image_id_set = list(set(train_triple[:, 2]))
        fix_size_index0=np.ones((batche_size))
        fix_size_index1=np.ones((batche_size))
        fix_size_index2=np.ones((batche_size))
        for i in range(batche_size):
            fix_size_index0[i] =  User_id_set.index(train_triple[i, 0])
            fix_size_index1[i] =  good_image_id_set.index(train_triple[i, 1])
            fix_size_index2[i] =  bed_image_id_set.index(train_triple[i, 2])
        # id去重之后进行可还原排序
        sort_index0 = np.argsort(User_id_set)
        sort_back_index0 = np.argsort(sort_index0)
        sort_index1 = np.argsort(good_image_id_set)
        sort_back_index1 = np.argsort(sort_index1)
        sort_index2 = np.argsort(bed_image_id_set)
        sort_back_index2 = np.argsort(sort_index2)
        sort_User_id_set = list(np.array(User_id_set)[sort_index0])
        sort_good_image_id_set = list(np.array(good_image_id_set)[sort_index1])
        sort_bed_image_id_set =  list(np.array(bed_image_id_set)[sort_index2])
        # 由去重并且排序之后的id序列读取数据
        input_User_vec = user_Data_file['userData'][sort_User_id_set]
        input_good_image = img_Data_file['imgData'][sort_good_image_id_set]
        input_bed_image =img_Data_file['imgData'][sort_bed_image_id_set]
        # 读取到的数据进行顺序还原
        unsort_input_User_vec = input_User_vec[sort_back_index0]
        unsort_input_good_image = input_good_image[sort_back_index1]
        unsort_input_bed_image = input_bed_image[sort_back_index2]
        # 读取到的数据进行去重还原
        fix_size_unsort_input_User_vec   = unsort_input_User_vec[fix_size_index0.astype(np.int32)]
        fix_size_unsort_input_good_image = unsort_input_good_image[fix_size_index1.astype(np.int32)]
        fix_size_unsort_input_bed_image  = unsort_input_bed_image[fix_size_index2.astype(np.int32)]
        Y = np.ones((batche_size))
        #返回数据
        yield ([fix_size_unsort_input_good_image, fix_size_unsort_input_bed_image,fix_size_unsort_input_User_vec], Y)
        batche = batche + 1
"""
定义验证&测试数据生成器
"""
def generate_test_sequences(batche,user_Data_file,img_Data_file,test_array_file,good_image_batche_size,bed_image_batche_size,user_batche_size = 1):
    # generate batches of samples
    user_id = test_array_file['test_array'][batche, 0]
    good_image_id = test_array_file['test_array'][batche, 1:good_image_batche_size+1]
    bed_image_id = test_array_file['test_array'][batche, good_image_batche_size+1:good_image_batche_size+bed_image_batche_size+1]
    # id集合可还原去重
    User_id_set = list(set([user_id]))
    good_image_id_set = list(set(good_image_id))
    bed_image_id_set = list(set(bed_image_id))

    fix_size_index0 = np.ones((user_batche_size))
    fix_size_index1 = np.ones((good_image_batche_size))
    fix_size_index2 = np.ones((bed_image_batche_size))
    for i in range(user_batche_size):
        fix_size_index0[i] = User_id_set.index([user_id][i])
    for i in range(good_image_batche_size):
        fix_size_index1[i] = good_image_id_set.index(good_image_id[i])
    for i in range(bed_image_batche_size):
        fix_size_index2[i] = bed_image_id_set.index(bed_image_id[i])
    # id去重之后进行可还原排序
    sort_index0 = np.argsort(User_id_set)
    sort_back_index0 = np.argsort(sort_index0)
    sort_index1 = np.argsort(good_image_id_set)
    sort_back_index1 = np.argsort(sort_index1)
    sort_index2 = np.argsort(bed_image_id_set)
    sort_back_index2 = np.argsort(sort_index2)
    sort_User_id_set = list(np.array(User_id_set)[sort_index0])
    sort_good_image_id_set = list(np.array(good_image_id_set)[sort_index1])
    sort_bed_image_id_set = list(np.array(bed_image_id_set)[sort_index2])
    # 由去重并且排序之后的id序列读取数据
    input_User_vec = user_Data_file['userData'][sort_User_id_set]
    input_good_image = img_Data_file['imgData'][sort_good_image_id_set]
    input_bed_image = img_Data_file['imgData'][sort_bed_image_id_set]
    # 读取到的数据进行顺序还原
    unsort_input_User_vec = input_User_vec[sort_back_index0]
    unsort_input_good_image = input_good_image[sort_back_index1]
    unsort_input_bed_image = input_bed_image[sort_back_index2]
    # 读取到的数据进行去重还原
    fix_size_unsort_input_User_vec = unsort_input_User_vec[fix_size_index0.astype(np.int32)]
    fix_size_unsort_input_good_image = unsort_input_good_image[fix_size_index1.astype(np.int32)]
    fix_size_unsort_input_bed_image = unsort_input_bed_image[fix_size_index2.astype(np.int32)]
    # 对齐数据
    image_data = np.zeros((good_image_batche_size + bed_image_batche_size, 256, 256, 3))
    user_data = np.ones((good_image_batche_size + bed_image_batche_size, 1024))
    image_data[0:good_image_batche_size, :] = fix_size_unsort_input_good_image
    image_data[good_image_batche_size:good_image_batche_size + bed_image_batche_size,:] = fix_size_unsort_input_bed_image
    for i in range(good_image_batche_size + bed_image_batche_size):
        user_data[i, :] = fix_size_unsort_input_User_vec
    return image_data, user_data
"""
自定义损失函数
"""
def CML_crossentropy(y_true, y_pred):

    # y_pred = K.exp(y_pred)
    # y_pred = y_pred / (1 + y_pred)
    # # return K.mean( K.minimum(-1 * K.log(y_pred), 100.), axis=-1)
    return K.mean(K.abs( y_pred - y_true), axis=-1)

"""
自定义评估函数
"""
def CML_top_k_accracy(y_true, y_pred):

    pass
    #return K.mean(K.log(y_pred), axis=-1)

if __name__ == '__main__':

    """
        ==============================编译模型==========================================
    """
    model = Comparative_Deep_Learning_of_Hybrid_Representations_network()
    model.compile(loss=CML_crossentropy, optimizer='SGD')
    #添加自定义评估函数
    # model.compile(loss=CML_crossentropy, optimizer='SGD', metrics=[CML_top_k_accracy])
    model.summary()

    """
        =================================训练===========================================
    """
    print model.get_weights()
    print "开始训练----"

    # generate batches of samples
    # 训练数据集大小 (2530186, 3)

    train_samples_size = 2530186
    train_batch_size = 256
    per_epoch_steps_sum = 300     #9883

    epochs = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    train_triple_path = '/home/shicheng/code/DataSet/flickr/train_triple_let.HDF5'
    train_triple_file = h5py.File(train_triple_path, 'r')

    user_Data_path = '/home/shicheng/code/DataSet/flickr/userData.HDF5'
    user_Data_file = h5py.File(user_Data_path, 'r')

    img_Data_path = '/home/shicheng/code/DataSet/flickr/imgData.HDF5'
    img_Data_file = h5py.File(img_Data_path, 'r')

    train_generator = generate_trian_sequences(train_triple_file, user_Data_file, img_Data_file, per_epoch_steps = per_epoch_steps_sum,batche_size = train_batch_size)
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=per_epoch_steps_sum,
                        epochs=epochs,
                        verbose=1,
                        metrics=[metrics.categorical_accuracy]
                        )

    train_triple_file.close()
    user_Data_file.close()
    img_Data_file.close()

    #保存模型参数
    print model.get_weights()
    model.save_weights('/home/shicheng/code/DataSet/flickr/my_model_weights_for_one_epochs.HDF5')

    """
        =================================测试===========================================
    """
    print "开始测试----"

    # test_array.HDF5
    # 测试数据集大小 (39920, 101)

    test_array_path = '/home/shicheng/code/DataSet/flickr/test_array.HDF5'
    test_array_file = h5py.File(test_array_path, 'r')

    # loss, accuracy = CML_model.evaluate(X_test,y_test)
    # print "\n测试误差",loss
    # print "top_k准确率",accuracy


    # generate_validation_sequences
    # test_triple_path = '/home/SSD/flickr/test_triple.hdf5'
    # with h5py.File(test_triple_path, 'r') as  test_triple :
    #
    #     images = test_file['feature']
    #     labels = test_file['labels']
    #
    #     idxs = range(len(images))
    #     test_idxs = idxs[: int(len(images))]
    #
    #     # testing sample generator
    #     n_test_batches = len(test_idxs) // batch_size
    #     n_remainder = len(test_idxs) % batch_size
    #     if n_remainder:
    #         n_test_batches = n_test_batches + 1
    #
    #     path2 = '/home/fanghuidi/aes-cnn/amyCNN/fusion/data/224-aes/feature_test.hdf5'
    #     with h5py.File(path2, 'r') as test_file2:
    #         images2 = test_file2['feature']
    #
    #         test_generator = generate_sequences(n_test_batches, images, images2, labels, test_idxs, (224, 224, 3))
    #
    #         preDistribution = model.predict_generator(generator=test_generator, steps=n_test_batches, verbose=1)
    #         sio.savemat('/home/fanghuidi/aes-cnn/amyCNN/fusion/results/predictions/preDistribution.mat',
    #                     {'preDistribution': preDistribution})
    test_array_file.close()


#--------------------------------xunlian--------------------------------------------
# CML_model.fit(X_train, y_train, epochs=1, batch_size=64,)
# fits the model on batches with real-time data augmentation:
# CML_model.fit_generator(data_generator.flow(x_train, y_train, batch_size=32),
#                         steps_per_epoch=len(x_train), epochs=epochs)
# CML_model.fit_generator(
#     train_generator,
#     # steps_per_epoch=2000 ,#// batch_size,
#     # epochs=50,
#     samples_per_epoch=9150,  # // batch_size,
#     nb_epoch=500,
#     validation_data=validation_generator,
#     nb_val_samples=1062,
#     callbacks=[checkpointer, history, plotter]
# )  # // batch_size)validation_steps=800
#
# # x_data = HDF5Matrix('input/file.hdf5', 'data')
# # model.predict(x_data)
#
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# y_train = np_utils.to_categorical(y_train, num_classes)
# y_test = np_utils.to_categorical(y_test, num_classes)
# epochs = 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# data_format = 'channels_last'

# iteration = 40
# validation_ratio = 0.1
# count = 0
# flag = 1
# best = 100
# trainLoss = np.zeros((iteration * 2), np.float32)
# valLoss = np.zeros((iteration * 2), np.float32)
# all_train = np.zeros((iteration), np.float32)
# all_val = np.zeros((iteration), np.float32)
# for i in range(iteration):
#     print 'epoch: {0}'.format(i)
#     history = model.fit_generator(generator = train_generator,
#                                    #validation_data=validation_generator,
#                                    steps_per_epoch=n_train_batches,
#                                    validation_steps=n_validation_batches1,
#                                    epochs=epochs)
#     val_loss = history1.history['val_loss'][0]
#     trainLoss[i * 2 + 0] = history1.history['loss'][0]
#     valLoss[i * 2 + 0] = history1.history['val_loss'][0]
#     all_train[i] = history1.history['loss'][0]
#     all_val[i] = val_loss
#
#     if (val_loss < best):
#         model.save_weights('/home/fanghuidi/aes-cnn/amyCNN/data/weights/weight.h5')
#         print 'imporve from {0} to {1}'.format(best, val_loss)
#         best = val_loss
#         count = 0
#         flag = 1
#
# plt.plot(all_train[:i])
# plt.plot(all_val[:i])
# plt.title('model loss at each epoch')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'])
# plt.grid()
# plt.show()
#--------------------------------ceshi--------------------------------------------


# print "开始测试----"
# loss, accuracy = CML_model.evaluate(X_test,y_test)
# print "\n测试误差",loss
# print "准确率",accuracy


# test_array.HDF5
# 测试数据集大小 (39920, 101)
# test_array_path = '/home/SSD/flickr/test_array.HDF5'
# test_array_file = h5py.File(test_array_path, 'r')

# generate batches of samples
# 三元组总个数 : 2530186
# per_epoch_steps = 2530186/batch_size ~~500次


# generate_validation_sequences
# test_triple_path = '/home/SSD/flickr/test_triple.hdf5'
# with h5py.File(test_triple_path, 'r') as  test_triple :
#
#     images = test_file['feature']
#     labels = test_file['labels']
#
#     idxs = range(len(images))
#     test_idxs = idxs[: int(len(images))]
#
#     # testing sample generator
#     n_test_batches = len(test_idxs) // batch_size
#     n_remainder = len(test_idxs) % batch_size
#     if n_remainder:
#         n_test_batches = n_test_batches + 1
#
#     path2 = '/home/fanghuidi/aes-cnn/amyCNN/fusion/data/224-aes/feature_test.hdf5'
#     with h5py.File(path2, 'r') as test_file2:
#         images2 = test_file2['feature']
#
#         test_generator = generate_sequences(n_test_batches, images, images2, labels, test_idxs, (224, 224, 3))
#
#         preDistribution = model.predict_generator(generator=test_generator, steps=n_test_batches, verbose=1)
#         sio.savemat('/home/fanghuidi/aes-cnn/amyCNN/fusion/results/predictions/preDistribution.mat',
#                    {'preDistribution': preDistribution})
# test_array_file.close()

















#
# def preprocess_image(image_path):
#     # 加载图像
#     img = image.load_img(image_path, target_size=(256, 256))
#     # 图像预处理
#     x = image.img_to_array(img)
#     return img
#
#
# #ImageDataGenerator用以生成一个batch的图像数据，支持实时数据提升。训练时该函数会无限生成数据，直到达到规定的epoch次数为止
# data_generator = ImageDataGenerator(
#     featurewise_center=True,                #使输入数据集去中心化（均值为0）
#     featurewise_std_normalization=True,     #将输入除以数据集的标准差以完成标准化
#     rotation_range=20,                      #数据提升时图片随机转动的角度
#     width_shift_range=0.2,                  #图片宽度的某个比例，数据提升时图片水平偏移的幅度
#     height_shift_range=0.2,                 #图片高度的某个比例，数据提升时图片竖直偏移的幅度
#     #rescale=1. / 255,
#     #shear_range=0.1,
#     #zoom_range=0.1,
#     horizontal_flip=True,                   #进行随机水平翻转
#     #fill_mode='nearest'
# )
#
# train_generator = data_generator.flow_from_directory(
#         r'chars_rec\train',  # this is the target directory
#         target_size=(32, 32),  # all images will be resized to 150x150
#         batch_size=32,
#         shuffle=True,
#         class_mode='categorical', color_mode='grayscale')  # since we use binary_crossentropy loss, we need binary labels
#
# # compute quantities required for featurewise normalization
# # (std, mean, and principal components if ZCA whitening is applied)
# data_generator.fit(x_train) #使用到featurewise的参数，则须传入整体数据计算参数，x：个数*长*宽*高的numpy array
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # here's a more "manual" example
# for e in range(epochs):
#     print 'Epoch', e
#     batches = 0
#     for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
#         loss = model.train(x_batch, y_batch)
#         batches += 1
#         if batches >= len(x_train) / 32:
#             # we need to break the loop by hand because
#             # the generator loops indefinitely
#             break
#
#





