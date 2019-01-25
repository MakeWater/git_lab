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

from get_pairs import get_pairs_by_None,get_pairs_by_siamese
from network import siamese
from utils import NMI,batch_generator,deepnn,predict_similarity,contro_loss,contrastive_loss,mnist_model

# 超参数：
params = {'n_clusters':10, 'n_nbrs':27, 'affinity':'nearest_neighbors'}
total_game_epoch = 2
epoch_train = 1
epoch_val = 10
# batch_size = 128

if not os.path.exists('log'):
    os.mkdir('log') 
log_dir = 'log' # 如果log目录不存在于当前目录，则在当前文件夹创建一个“log”目录并赋值给log_dir对象

# unlabel_data, _ = get_data()
# mnist_data = np.load('mnist.npy').astype(np.float32)
mnist = input_data.read_data_sets('MNIST_data',one_hot=False)
# label = np.load('mnist_lab.npy')[:1000] # for NMI computation
# unlabel_data = mnist_data[:1000]

# test_100_data = np.load('test_100_data.npy')
test_100 = np.load('test_100.npy')
pairs = np.load('pairs.npy').astype(np.float32)
pairs_label = np.load('pairs_label.npy').astype(np.float32)
sess = tf.InteractiveSession()
siam = siamese()

# # inputs:
# left = tf.placeholder(tf.float32,[None,784],name='left_input')
# right = tf.placeholder(tf.float32,[None,784],name='right_input')
# y_ = tf.placeholder(tf.float32,[None,1],name='target_similarity_of_pairs')
  
# left_output = deepnn(left)
# right_output = deepnn(right)
# simi = predict_similarity(left_output,right_output)
# loss = tf.losses.cosine_distance(simi,y_,axis=0)
# loss = contro_loss(left_output,right_output,y_)
# loss = contrastive_loss(left_output,right_output,y_,margin=0.5)

for batch_size in [32,64,128,512,1024]:
    global_step = tf.Variable(0,trainable=False) #只有变量（variable）才要初始化，张量（Tensor）是没法初始化的
    with tf.name_scope('learning_rate'):
        learning_rate_0 = tf.Variable(0.09,name='initial_lr')
        learning_rate_decay_steps = 0.5*len(pairs)/batch_size
        learning_rate = tf.train.exponential_decay(learning_rate_0,global_step,learning_rate_decay_steps,0.96) # 每喂入100个batch_size的数据后学习率衰减到最近一次的96%。
        # tf.summary.scalar('learning_rate',learning_rate)

    # train_writer = tf.summary.FileWriter(log_dir + '/train',sess.graph)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(siam.loss,global_step=global_step) # train_step是一个‘operation’对象，不能初始化
    # 初始化全部变量，包括网络、学习率、global_step等
    sess.run(tf.global_variables_initializer())

    for game_epoch in range(total_game_epoch):
        if game_epoch == 0:
            # merged = tf.summary.merge_all()
            # pairs, pairs_label, class_indices, index_to_pair, label_pred = get_pairs_by_None(unlabel_data,params)

            # pairs = pairs[:1000]
            # pairs_label = pairs_label[:1000]
            # NMI_score = NMI(label_pred,label)
            # print('the mean NMI score is:',NMI_score)
            # print('pairs shape is:{},pairs label shape is:{}'.format(pairs.shape[0],pairs_label.shape[0]))

            # np.save('pairs.npy',pairs)
            # np.save('pairs_label.npy',pairs_label)

            for epoch in range(epoch_train):
                # there should be shuffle each epoch.
                # sess.run([learning_rate_0.initializer,global_step.initializer])
                # shuffle = np.random.permutation(pairs.shape[0])
                # pairs = pairs[shuffle]
                # pairs_label = pairs_label[shuffle]
                print('The next epoch is ',epoch)
                # shuffle the pairs each epoch
                # batch_loss_list = []
                # data_generator = batch_generator(pairs,pairs_label,batch_size)
                # steps = 0
                # 这个for循环只是用来读取数据的。 每取一个batch的数据就是一个step
                for steps in range(100000):# get batch data from data generator
                    x1,y1 = mnist.train.next_batch(batch_size)
                    x2,y2 = mnist.train.next_batch(batch_size)
                    y_true = (y1==y2).astype('float')
                    # y_true = np.expand_dims(y_true,-1)
                    _, losses = sess.run([train_step,siam.loss], 
                                                            feed_dict={
                                                                        siam.x1: x1,
                                                                        siam.x2: x2,
                                                                        siam.y_true: y_true})
                    # batch_loss_list.append(np.array(losses))
                    if steps%100==0:

                        mean_loss = np.mean(losses)
                        # train_writer.add_summary(summary,steps)
                        # saver.save(sess,os.path.join(log_dir + 'model','model.ckpt'),steps)# save trained model.
                        print('the game_epoch is %d,epoch is %d,step is %d, batch_size is %d, mean loss is %.3f,current learning_rate is %.8f, branch is %s' % 
                            (game_epoch, epoch, steps, batch_size, mean_loss,sess.run(learning_rate),'Master'))
                    # steps += 1
            # train_writer.close()


        #################################################### 新的一个循环 ###################################################

        elif game_epoch > 0:
            # 1、准备数据：W，pairs，pairs label, 修改params
            # compute the affinity matrix by siamese:

            # n = len(unlabel_data)
            # W = np.zeros((n,n))
            # start = time.clock()
            # for i in range(n):
            #     for j in range(i,n):
            #         if i == j:
            #             W[i][j] = 0.5
            #         else:
            #             # approch 2:(solve the shape problem by array = np.expand_dims(array,axis=0)
            #             W[i][j] = sess.run(siam.similarity,
            #                 feed_dict={siam.x1:np.expand_dims(unlabel_data[i],axis=0),
            #                            siam.x2:np.expand_dims(unlabel_data[j],axis=0)})
            #             if j%100==1:
            #                 print('the similarity between {} and {} is {}'.format(i,j,W[i][j]))
            # elapsed = (time.clock() - start)
            # print('Time used to compute affinity :',elapsed)

            # np.save('W_{}_0.npy'.format(game_epoch),W) # 转置前的W 
            # # 转置成对称阵
            # W = W + W.transpose()
            # np.save('W_{}.npy'.format(game_epoch),W) # W 转为对称阵 

            W_test = np.zeros((100,100),dtype=np.float64)
            for i in range(100):
                for j in range(100):
                    if i==j:
                        W_test[i][j] = 1
                    else:
                        W_test[i][j] = sess.run(siam.distance,
                                feed_dict={siam.x1:np.expand_dims(test_100[i], axis=0),
                                           siam.x2:np.expand_dims(test_100[j], axis=0)})
            max_distance = np.max(W_test)
            W_test = max_distance - W_test
            np.save('W_test_batch_size{}_Master.npy'.format(batch_size),W_test)
            print('AFFINITY HAS BEEN COMPUTED AND SAVED ! ##########################################################')

        '''
        # 预测新的对
        # 这里的class_indices可以验证谱聚类的效果,尤其是purf_idx,其实是纯化后的索引。
        pairs1, pairs_label_1, class_indices, index_to_pair, label_pred = get_pairs_by_siamese(unlabel_data,W,params) # 验证新矩阵的效果。
        # save pairs data:
        np.save('pairs{}.npy'.format(game_epoch),pairs1)
        np.save('pairs_label_{}.npy'.format(game_epoch),pairs_label_1)
        np.save('index_to_pair{}.npy'.format(game_epoch),index_to_pair)
        # get NMI score: 

        NMI_score = NMI(label_pred,label)
        print('the mean NMI score is:',NMI_score)
        # if purf_idx.shape[0] < 10:
        #     cluster_num = 10
        #     params['n_clusters'] = cluster_num
        # params['n_clusters'] = purf_idx.shape[0]
        # print('pairs1 shape is:%d,pairs1 label shape is:%d' % (pairs1.shape,pairs_label_1.shape))
        #本质上是用上一次网络计算的矩阵预测的对来进行下一次的网络训练：

        global_step = tf.Variable(0,trainable=False)
        with tf.name_scope('learning_rate'):
            learning_rate_0 = tf.Variable(0.01,name='initial_lr')
            learning_rate = tf.train.exponential_decay(learning_rate_0,global_step,500,0.95)
            # tf.summary.scalar('learning_rate',learning_rate)

        sess.run([learning_rate_0.initializer,global_step.initializer])
        #或者运行：
        # sess.run(tf.variables_initializer([learning_rate_0,global_step]))

        # merged = tf.summary.merge_all()
        # test_writer = tf.summary.FileWriter(log_dir + '/test{}'.format(game_epoch))
        shuffle = np.random.permutation(pairs1.shape[0])
        pairs1 = pairs1[shuffle]
        pairs_label_1 = pairs_label_1[shuffle]
        print('game_epoch',game_epoch)
        for batch_size in [8,16,32,64]:
            for epoch2 in range(epoch_val):
                batch_loss_list = []
                data_generator2 = batch_generator(pairs1,pairs_label_1,batch_size)
                step_2 = 0
                # 这个for循环只是用来读取数据的。
                for ([batch_x1,batch_x2],y_true) in data_generator2:
                    x1 = batch_x1
                    x2 = batch_x2
                    y_true = np.expand_dims(y_true,-1)

                # for i in range(pairs1.shape[0]):
                #     # 逐对的喂入数据：
                #     x1 = np.expand_dims(pairs1[i][0],axis=0)
                #     x2 = np.expand_dims(pairs1[i][1],axis=0)
                #     y_true = np.expand_dims(pairs_label_1[i],axis=0)
                    
                    _, losses= sess.run([train_step,loss], 
                                                        feed_dict={
                                                                    siam.x1: x1,
                                                                    siam.x2: x2,
                                                                    siam.y_true: y_true})
                    batch_loss_list.append(losses)
                    if step_2 % 100 == 0:
                        mean_loss = np.mean(batch_loss_list)
                        # test_writer.add_summary(summary,step_2)
                        # saver.save(sess,os.path.join(log_dir + 'model','model.ckpt'),steps)# save trained model.
                        print('the game_epoch is %d,epoch is %d,step is %d, batch_size is %d, mean loss is %.3f' % 
                            (game_epoch, epoch2, step_2, batch_size, mean_loss))
                    step_2 += 1
        '''
print('Done Training!')