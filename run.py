# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os
import math
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
from sklearn import cluster

from get_pairs import get_pairs_by_None,get_pairs_by_siamese,display_label,spectral_clustering,get_class_indices,creat_sample_pair
from network import siamese
from utils import NMI,batch_generator


random.seed(0)
# 超参数：
params = {'n_clusters':2, 'n_neighbors':7,'n_init':20, 'affinity':'rbf','gamma':1.0,'n_jobs':-1}
# params = {'n_clusters':10, 'n_nbrs':27, 'affinity':'nearest_neighbors'}
total_game_epoch = 5
epoch_train = 10
epoch_val = 5

# if not os.path.exists('log'):
#     os.mkdir('log') 
# log_dir = 'log' # 如果log目录不存在于当前目录，则在当前文件夹创建一个“log”目录并赋值给log_dir对象

mnist_data = np.load('mnist.npy').astype(np.float32)
unlabel_data = mnist_data[0:1000]
# mnist = input_data.read_data_sets('MNIST_data',one_hot=False)
label = np.load('mnist_lab.npy')[0:1000] # for NMI computation

# test_100_data = np.load('test_100_data.npy')
test_100 = np.load('test_100.npy')

# pairs = np.load('pairs.npy').astype(np.float32)
# pairs_label = np.load('pairs_label.npy').astype(np.float32)

sess = tf.InteractiveSession()
siam = siamese()

# pairs, pairs_label, class_indices, index_to_pair, label_pred = get_pairs_by_None(unlabel_data,params,use_autoencoder=False,num_to_del=3)
# pairs = np.load('sample_pairs.npy')
# pairs_label = np.load('sample_pairs_label.npy')
pairs = np.load('sample_pairs.npy')
pairs_label = np.load('sample_pairs_label.npy')
print("pairs shape is:",pairs.shape)
print("pairs_label shape is :",pairs_label.shape)
shuffle = np.random.permutation(pairs.shape[0])
pairs = pairs[shuffle]
pairs_label = pairs_label[shuffle]
# exit()

for batch_size in [8]:
    global_step = tf.Variable(0,trainable=False) #只有变量（variable）才要初始化，张量（Tensor）是没法初始化的
    with tf.name_scope('learning_rate'):
        learning_rate_0 = tf.Variable(0.01,name='initial_lr')
        learning_rate_decay_steps = 0.5*len(pairs)/batch_size
        learning_rate = tf.train.exponential_decay(learning_rate_0,global_step,learning_rate_decay_steps,0.96) # 每喂入100个batch_size的数据后学习率衰减到最近一次的96%。
        # tf.summary.scalar('learning_rate',learning_rate)

    # train_writer = tf.summary.FileWriter(log_dir + '/train',sess.graph)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(siam.loss,global_step=global_step) # train_step是一个‘operation’对象，不能初始化
    # 初始化全部变量，包括网络、学习率、global_step等
    sess.run(tf.global_variables_initializer())

    for game_epoch in range(total_game_epoch):
        # if game_epoch == 0:
        #     # merged = tf.summary.merge_all()
        # +++++++++++++++ ###
        # train siamese net:
        for epoch in range(epoch_train):
            # sess.run([learning_rate_0.initializer,global_step.initializer])
            print('The next epoch is ',epoch)
            data_generator = batch_generator(pairs,pairs_label,batch_size)
            steps = 0
            for ([x1,x2],y_true) in data_generator:
                _, losses = sess.run([train_step,siam.loss], 
                                                        feed_dict={
                                                                    siam.x1: x1,
                                                                    siam.x2: x2,
                                                                    siam.y_true: y_true})
                if steps%100==0:
                    mean_loss = np.mean(losses)
                    # train_writer.add_summary(summary,steps)
                    # saver.save(sess,os.path.join(log_dir + 'model','model.ckpt'),steps)# save trained model.
                    print('the game_epoch is %d,epoch is %d,step is %d, batch_size is %d, loss is %.3f,current learning_rate is %.8f, branch is %s' % 
                        (game_epoch, epoch, steps, batch_size, mean_loss,sess.run(learning_rate),'Master'))
                steps += 1

        # compute matrix W
        val_data = np.load('val_data.npy')
        n = len(val_data)
        W = np.zeros((n,n))
        start = time.clock()
        for i in range(n):
            for j in range(n):
                if i == j:
                    W[i][j] = 0
                else:
                    W[i][j] = sess.run(siam.distance,
                        feed_dict={siam.x1:np.expand_dims(val_data[i],axis=0),
                                   siam.x2:np.expand_dims(val_data[j],axis=0)})
                    if j%100==1:
                        print('the distance between {} and {} is {}'.format(i,j,W[i][j]))
        elapsed = (time.clock() - start)
        print('Time used to compute affinity :',elapsed)
        pred_label_by_SC = spectral_clustering(W,params)
        class_idx = get_class_indices(pred_label_by_SC)
        np.save('class_idx.npy',class_idx)
        val_label = np.load('val_label.npy')
        display_label(class_idx,val_label)
        pairs,pairs_label= creat_sample_pair(val_data,val_label,class_idx)

'''
        #################################################### 新的一个循环 ###################################################
        elif game_epoch > 0:
            # 孪生网络第一次训练完后：
            # 1、准备数据：W，pairs，pairs_label, 修改params中聚类的目标簇数、矩阵的来源
            # 2、使用第一次训练的孪生网络计算出矩阵W：
            val_data = np.load('val_data.npy')
            n = len(val_data)
            W = np.zeros((n,n))
            start = time.clock()
            for i in range(n):
                for j in range(n):
                    if i == j:
                        W[i][j] = 0
                    else:
                        # approch 2:(solve the shape problem by array = np.expand_dims(array,axis=0)
                        W[i][j] = sess.run(siam.distance,
                            feed_dict={siam.x1:np.expand_dims(val_data[i],axis=0),
                                       siam.x2:np.expand_dims(val_data[j],axis=0)})
                        if j%100==1:
                            print('the distance between {} and {} is {}'.format(i,j,W[i][j]))
            elapsed = (time.clock() - start)
            print('Time used to compute affinity :',elapsed)
            # np.save('W_{}_0.npy'.format(game_epoch),W) # 转置前的W 
            # 转置成对称阵
            # W = W + W.transpose()
            # max_distance = np.max(W)
            # W = W / float(max_distance)
            # W = 1.0 - W
            # np.save('W_{}.npy'.format(game_epoch),W)
            
            pred_label_by_SC = spectral_clustering(W,params)
            class_idx = get_class_indices(pred_label_by_SC)
            np.save('class_idx.npy',class_idx)
            val_label = np.load('val_label.npy')
            display_label(class_idx,val_label)
            pairs,pairs_label= creat_sample_pair(val_data,val_label,class_idx)


            # exit()

            # W_test = np.zeros((100,100),dtype=np.float64)
            # for i in range(100):
            #     for j in range(100):
            #         if i==j:
            #             W_test[i][j] = 0
            #         else:
            #             W_test[i][j] = sess.run(siam.distance,
            #                     feed_dict={siam.x1:np.expand_dims(test_100[i], axis=0),
            #                                siam.x2:np.expand_dims(test_100[j], axis=0)})
            # max_distance = np.max(W_test)
            # W_test = W_test / float(max_distance)
            # W_test = 1.0 - W_test
            # np.save('W_test_batch_size{}_Master.npy'.format(game_epoch),W_test)
            print('AFFINITY HAS BEEN COMPUTED AND SAVED ! ##########################################################')

            # 预测新的对
            # 这里的class_indices可以验证谱聚类的效果,尤其是purf_idx,其实是纯化后的索引。
            # 这里W是孪生网络计算得到的，param中的一些参数要改了：n_clusters,affinity


            
            params['n_clusters'] = 10
            pairs1, pairs_label_1, class_indices, index_to_pair, label_pred = get_pairs_by_siamese(unlabel_data,W,params,num_to_del=0) # 验证新矩阵的效果。
            display_label(index_to_pair,label)
            print('pairs1 shape is:{},pairs1 label shape is:{}'.format(pairs1.shape,pairs_label_1.shape))
            np.save('index_to_pair{}.npy'.format(game_epoch),index_to_pair)

            NMI_score = NMI(label_pred,label)
            print('the mean NMI score of epoch{} is:'.format(game_epoch),NMI_score)

            global_step = tf.Variable(0,trainable=False)
            with tf.name_scope('learning_rate'):
                learning_rate_0 = tf.Variable(0.01,name='initial_lr')
                learning_rate_decay_steps = 0.5*len(pairs)/batch_size
                learning_rate = tf.train.exponential_decay(learning_rate_0,global_step,learning_rate_decay_steps,0.95)
                # tf.summary.scalar('learning_rate',learning_rate)

            sess.run([learning_rate_0.initializer,global_step.initializer])
            # sess.run(tf.variables_initializer([learning_rate_0,global_step]))
            # test_writer = tf.summary.FileWriter(log_dir + '/test{}'.format(game_epoch))
            shuffle = np.random.permutation(pairs1.shape[0])
            pairs1 = pairs1[shuffle]
            pairs_label_1 = pairs_label_1[shuffle]
            





            print('game_epoch',game_epoch)
            for epoch2 in range(epoch_val):
                batch_loss_list = []
                data_generator2 = batch_generator(pairs,pairs_label,batch_size)
                step_2 = 0
                # 这个for循环只是用来读取数据的。
                for ([batch_x1,batch_x2],y_true) in data_generator2:
                    x1 = batch_x1
                    x2 = batch_x2             
                    _, losses= sess.run([train_step,siam.loss], 
                                                        feed_dict={
                                                                    siam.x1: x1,
                                                                    siam.x2: x2,
                                                                    siam.y_true: y_true})
                    # batch_loss_list.append(losses)
                    if step_2 % 100 == 0:
                        mean_loss = np.mean(losses)
                        # test_writer.add_summary(summary,step_2)
                        # saver.save(sess,os.path.join(log_dir + 'model','model.ckpt'),steps)# save trained model.
                        print('the game_epoch is %d,epoch is %d,step is %d, batch_size is %d, mean loss is %.3f' % 
                            (game_epoch, epoch2, step_2, batch_size, mean_loss))
                    step_2 += 1


print('Done Training!')

'''