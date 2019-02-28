# -*-coding:utf-8 -*-
# -*- coding:utf-8 -*-
import numpy as np 
import random
import cv2
import collections 
from collections import defaultdict
from sklearn import cluster
from operator import itemgetter
from itertools import permutations,combinations
from keras.datasets import mnist
from keras.layers import Input,Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.losses import mse, binary_crossentropy

def autoencoder(data):
    ''' 获取数据embedding
    data:data to be encoded,not used for train.
    return: embeded code of data
    '''
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    data_train = np.load('mnist.npy').astype('float32')/255.0 #(70000,784)

    # network parameters
    input_shape = (784, ) # 784
    intermediate_dim = 512
    batch_size = 128
    latent_dim = 10
    epochs = 50

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    x = Dense(2048,activation='relu')(x)
    encoded = Dense(latent_dim,activation='relu')(x)

    x = Dense(2048, activation='relu')(encoded)
    x = Dense(intermediate_dim, activation='relu')(x)
    decoded = Dense(784,activation='sigmoid')(x)
    # instantiate decoder model
    encoder = Model(inputs, encoded, name='encoder')
    autoencoder = Model(inputs, decoded, name='decoder')

    autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
    autoencoder.fit(data_train,data_train,
                    epochs=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test,x_test))
    autoencoder.save_weights('autoencoder_mnist.h5')
    # 对 raw data 进行嵌入工作。
    embeded = encoder.predict(data)
    np.save('embeded.npy',embeded)
    return embeded

def SC(W,params):
    ''' 
    Implementing spectral clustering algorithm on unlabel data(or embeded) 
    in shape(sample_nums,784) or affinity matrix W

    data: unlabel data or embeded data of unlabel data
    W : affinity matrix,is None or a precomputed affinity matrix.
    params: params applied to configrate spectral clustering algorithm
    return: predicted label.
    '''
    spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'],
        eigen_solver='arpack',affinity=params['affinity'],eigen_tol=params['n_clusters'],n_neighbors=params['n_nbrs'])
    s = spectral.fit(W)
    # np.save('SC_matrix.npy',s.affinity_matrix_)
    label_pred = s.fit_predict(W)
    return label_pred

def get_pairs(data,W,params):
    # data = np.load('mnist.npy')[:1000]
    # W = None
    # params = {'n_clusters':12, 'n_nbrs':27, 'affinity':'nearest_neighbors'}

    ''' get pairs according to label_pred by SC
        data: data. data prepare to get pairs.
        W: None or affinity W,if None,let W = data,defult value is None.
        return: pairs and pairs' label used to siamese network
    '''
    if W is not None:
        print('affnity is precomputed')
        params['affnity'] = 'precomputed'
        lab_pred = SC(W,params)
        pairs,pairs_lab,index_to_pair= creat_pairs(data,lab_pred)
    else:
        print('need to code the raw data before compute the affinity')
        embeded = autoencoder(data)
        np.save('embeded.npy',embeded)

        # embeded = np.load('embeded.npy')

        W = embeded # W is embeded data for the first time spectral clustering.
        lab_pred = SC(W,params)
        pairs,pairs_lab,index_to_pair = creat_pairs(data,lab_pred)
    # shuffle pairs and corresponding labels 这算是第二次打乱了，第一次在create_pairs的时候就已经打乱了
    index = np.random.permutation(pairs.shape[0])
    pairs = pairs[index]
    pairs_lab = pairs_lab[index]

    return pairs,pairs_lab,index_to_pair

def creat_pairs(data,label):
    '''data is raw data,label is predicted by spectral clustering(SC)
    params_sub = {'n_clusters':2, 'n_nbrs':len(dt)/2, 'affinity':'nearest_neighbors'}

    '''
    mnist_label = np.load('mnist_lab.npy')

    class_indices = get_class_indices(label)
    lenth_of_each_class = []
    for i in range(len(class_indices)):
        lenth_of_each_class.append(len(class_indices[i]))
    minimum_dim = min(lenth_of_each_class)
    class_indices_new = []
    for i in range(len(class_indices)):
        class_indices_new.append(class_indices[i][:minimum_dim])
    class_indices = np.array(class_indices_new)
    # class_indices = delete_mess_row(class_indices,1) # for mnist,only delete the largest cluster
    display_label(class_indices,mnist_label)
    # index_to_pair = strainer_of_classindices(data,class_indices)

    np.save('class_indices.npy',class_indices)
    # np.save('index_to_pair.npy',index_to_pair)

    print('######################## SPLIT LINE ######################################')
    # display label distribute
    # display_label(index_to_pair,mnist_label)
    # exit()
    cluster_number = len(class_indices)
    pos_pairs_temp = []
    # 对每一个样本进行旋转后进行相互配对，得到关于自身的配对。
    for i in range(cluster_number):
        pos_pair_generator = combinations(class_indices[i],2)      # index_to_pair >> class_indices
        single_pairs = [[data[idx1],data[idx2]] for (idx1,idx2) in pos_pair_generator] #同一类只在原数据间配对，它们的旋转之间不配对
        pos_pairs_temp.append(single_pairs)
        #---------------------------------------------# index_to_pair 每一行的数据所能产生的所有对
        #---------此次修改主要改进了配对方式使所配的对更准确。同时注意到permutations（）函数是排列，不是组合。-------------#
    
    pos_pairs = []
    for i in range(len(pos_pairs_temp)):
        for p_ in pos_pairs_temp[i]:
            pos_pairs.append(p_)

    neg_pairs = []
    neg_num = np.arange(len(pos_pairs))
    for _ in neg_num:
        c_1 = random.choice(np.arange(cluster_number)) # cluster 1
        inc = random.randrange(1,cluster_number)
        c_2 = (c_1+inc) % cluster_number # cluster 
        idx1 = random.choice(class_indices[c_1])
        idx2 = random.choice(class_indices[c_2])
        neg_pairs.append([data[idx1],data[idx2]])
    pos_pairs = np.array(pos_pairs)
    neg_pairs = np.array(neg_pairs)
    pos_lab = np.ones(len(pos_pairs))
    neg_lab = np.zeros(len(neg_pairs))
    pairs = np.concatenate((pos_pairs,neg_pairs),axis=0)
    pairs_lab = np.concatenate((pos_lab,neg_lab))
    shuffle = np.random.permutation(pairs.shape[0])
    pairs = pairs[shuffle]
    pairs_lab = pairs_lab[shuffle]
    print('pairs shape is:',pairs.shape)
    return pairs,pairs_lab,class_indices

def display_label(index_to_display,label_true):
    for i in range(len(index_to_display)):
        arr = np.array(index_to_display[i],dtype=np.int32)
        print([len(index_to_display[i]), label_true[arr]])

def strainer_of_classindices(data,class_indices):
    '''filter clusters to be more purity.
    data: raw mnist data in the shape (number,784)
    class_indices: primary assigned subcluster
    return: a purified class_indices use spectral clustering with new "params_sub"
    '''
    index_to_pair = []
    for i in range(len(class_indices)):
        dt = data[class_indices[i]] # subcluster data
        # dt_index = np.array([np.argwhere(dt==elem) for elem in dt]).reshape(-1)
        dt_index = np.arange(len(dt)) # 获取子cluster每一个数据样本的下标
        index_save = np.array(zip(dt_index,class_indices[i]))
        params_sub = {'n_clusters':2, 'n_nbrs':len(dt)/2, 'affinity':'nearest_neighbors'}
        lab_sub_pred = SC(dt,params_sub) 
        sub_class_indices = get_class_indices(lab_sub_pred) # index number from 0 ~ len(dt)
        cleaner_part = sub_class_indices[0] if len(sub_class_indices[0]) > len(sub_class_indices[1]) else sub_class_indices[1]
        purified_idx = index_save[cleaner_part,1] # purified data indices of each subcluster where it's in raw datasets.
        index_to_pair.append(purified_idx)
    return index_to_pair

def get_class_indices(label):
    categories = len(collections.Counter(label))
    class_indices = [np.where(label == i)[0] for i in range(categories)]
    class_indices = np.array(class_indices)
    return class_indices

def delete_mess_row(class_indices,num_to_del):
    '''
    class_indices:锯齿状的列表，每一行都是一类数据的索引
    这一步的目的在于谱聚类每一次聚类都会产生一类数量特别多但是分类又特别差的类。

    return:
           new_indices: np array deleted the mess class.
           mess_idx: python list with the longest mess data indices. 
                     We can use it as validation data.
    '''
    class_indices = np.array(class_indices)
    len_temp = []
    for i in range(len(class_indices)):
        len_temp.append((i,len(class_indices[i])))
    len_temp = sorted(len_temp,key=itemgetter(1),reverse=True)
    # delete the first several longest row in class_indices
    for i in range(num_to_del):
        class_indices = np.delete(class_indices,[len_temp[i][0]],axis=0)
    new_indices = class_indices
    return new_indices

def rot_img(img,angle):
    '''rotate image with crop
    img : a 2D matrix
    angle: a inter in 0~360 angle,counter-clockwise direction.
    return: a rotated image(croped) in give angle,in anti-clockwise direction.shape(28,28)
    '''
    img = img.reshape(28,28)
    rows,cols = img.shape[:2]
    roted_matrix = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    roted_img = cv2.warpAffine(img,roted_matrix,(cols,rows))
    return roted_img

# if __name__ == '__main__':
#     pairs,pairs_label,pairs_index = get_pairs()
#     np.save('pairs.npy',pairs),np.save('pairs_label.npy',pairs_label),np.save('pairs_index.npy',pairs_index)