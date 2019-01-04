# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np

''' Definite Siamese network as a measure function after trained by pairs data'''
class siamese():
    ''' siamese network two steps: train by pairs data and pairs label;
                                   measure similarity between samples as a measure function
    '''
    def __init__(self):

        with tf.name_scope('input'):
            with tf.name_scope('input_x1'):
                self.x1 = tf.placeholder(tf.float32,[None,784])
            with tf.name_scope('input_x2'):
                self.x2 = tf.placeholder(tf.float32,[None,784])
            with tf.name_scope('y_input'):
                self.y_true = tf.placeholder(tf.float32,[None])
            # with tf.name_scope('dropout'):
                # self.dropout = tf.placeholder(tf.float32)

        with tf.variable_scope('siamese') as scope:
            self.output1 = self.deepnn(self.x1) # shape:(1000,10) or (1,10)
            # scope.reuse_variables()
            self.output2 = self.deepnn(self.x2)
            with tf.name_scope('similarity'):
                self.similarity = self.predict_similarity(self.output1,self.output2)
        
        with tf.name_scope('loss'):
            self.loss = self.contro_loss(self.similarity,self.y_true)
            # tf.summary.scalar('loss',self.loss)

    def contro_loss(self,similarity_by_network,pairs_label):

        '''
        总结下来对比损失的特点：首先看标签，然后标签为1是正对，负对部分损失为0，最小化总损失就是最小化类内损失(within_loss)部分，
        让s逼近margin的过程，是个增大的过程；标签为0是负对，正对部分损失为0，最小化总损失就是最小化between_loss，而且此时between_loss就是s，
        所以这个过程也是最小化s的过程，也就使不相似的对更不相似了.
        最小化类内损失的是一个增大s的过程，最小化类间损失的是一个减少s的过程。
        '''
        
        s = similarity_by_network
        # one = tf.constant(1.0)
        margin = 1.0
        y_true = pairs_label

        # 类内损失：
        # max_part = tf.square(tf.maximum(margin-s,0)) # margin是一个正对该有的相似度临界值，如：1
        diff_part = margin - s 
        #如果相似度s未达到临界值margin，则最小化这个类内损失使s逼近这个margin，增大s
        within_loss = tf.multiply(y_true,diff_part) 

        # 类间损失：
        #如果是负对，between_loss就等于s，这时候within_loss=0，最小化损失就是降低相似度s使之更不相似
        between_loss = tf.multiply(1.0-y_true,s) 

        # 总体损失（要最小化）：
        loss = tf.reduce_mean(within_loss+between_loss) 
        return loss


    def deepnn(self,x):

        # input reshape to [batch_size,28,28,channel]
        with tf.name_scope('reshape'):
            # transform input type to tensorflow type
            # x = tf.cast(x,tf.float32)
            x_image = tf.reshape(x,[-1,28,28,1])
            tf.summary.image('input',x_image,10)

        # layer1: picture width = 28->28
        with tf.name_scope('conv1'):
            w_conv1 = self.weight_variable([5,5,1,32])
            self.variable_summaries(w_conv1)
            b_conv1 = self.bias_variable([32])
            self.variable_summaries(b_conv1)
            h_conv1 = tf.nn.relu(self.conv2d(x_image,w_conv1) + b_conv1)
            tf.summary.histogram('activation1',h_conv1)

        # pooling layer1 : 28->14
        with tf.name_scope('pooling1'):
            h_pooling_1 = self.max_pool_2x2(h_conv1)

        # convolution layer2 : 14->14
        with tf.name_scope('conv2'):
            w_conv2 = self.weight_variable([5,5,32,64])
            self.variable_summaries(w_conv2)
            b_conv2 = self.weight_variable([64])
            self.variable_summaries(b_conv2)
            h_conv2 = tf.nn.relu(self.conv2d(h_pooling_1,w_conv2) + b_conv2)
            tf.summary.histogram('activation2',h_conv2)

        # feature width : 14->7
        with tf.name_scope('pooling2'):
            h_pool2 = self.max_pool_2x2(h_conv2)

        with tf.name_scope('fc1'):
            # W2 = (W1-F+2P)/S + 1
            w_fc1 = self.weight_variable([7*7*64,1024])
            self.variable_summaries(w_fc1)
            b_fc1 = self.bias_variable([1024])
            self.variable_summaries(b_fc1)
            h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)
            tf.summary.histogram('activation3',h_fc1)

        # with tf.name_scope('dropout'):
            # h_fc1_drop = tf.nn.dropout(h_fc1,self.dropout)

        with tf.name_scope('fc2'):
            # embedding in shape: [batch_size,10]
            w_fc2 = self.weight_variable([1024,10])
            self.variable_summaries(w_fc2)
            b_fc2 = self.bias_variable([10])
            self.variable_summaries(b_fc2)
            embedding = tf.matmul(h_fc1,w_fc2)+b_fc2
            tf.summary.histogram('embedding',embedding)
            # embedding = tf.nn.softmax(embedding,dim=1)
            # tf.reshape(embedding,[1000,10])
        return embedding

 
    def predict_similarity(self,embedding1,embedding2):
        '''
        要注意的是当样本batch_size为1时，用这个当做函数来计算两两样本之间的相似度需要注意最后求均值的axis问题
        '''
        # A, B分别是两个样本经过网络传播之后的提取后的特征/embedding
        # 求两个向量的余弦夹角：A*B/|A|*|B|
        # 求每对样本之间的相似度，即使一个batch_size也是先求各自的再求平均
        cosi = tf.divide(
                        tf.reduce_sum(tf.multiply(embedding1,embedding2),axis=1,keep_dims=True),
                        tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(embedding1),axis=1,keep_dims=True)),
                                    tf.sqrt(tf.reduce_sum(tf.square(embedding2),axis=1,keep_dims=True))))
        cosi = (cosi+1.0)/2.0 # 平移伸缩变换到[0,1]区间内,谱聚类算法要求的亲和矩阵中不能产生负值。
        # cosi batch_size shape：（batch_size，1）
        return cosi



    def variable_summaries(self,var):
        '''Attach a lot of summaries to a Tensor (for Tensorboard visualization).'''
        with tf.name_scope('summaries'):
            # computing mean of var recording by tf.summary.scalar
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean',mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
            # record std,maximum,minimum
            tf.summary.scalar('stddev',stddev)
            tf.summary.scalar('max',tf.reduce_max(var))
            tf.summary.scalar('min',tf.reduce_min(var))
            # record the distribution of var in histogram
            tf.summary.histogram('histogram',var)

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x,w):
        return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
