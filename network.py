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
        # input
        self.x1 = tf.placeholder(tf.float32,[None,784])
        self.x2 = tf.placeholder(tf.float32,[None,784])
        self.y_true = tf.placeholder(tf.float32,[None])
            # with tf.name_scope('dropout'):
                # self.dropout = tf.placeholder(tf.float32)

        with tf.variable_scope('siamese') as scope:
            self.output1 = self.network(self.x1) # shape:(1000,10) or (1,10)
            scope.reuse_variables()
            self.output2 = self.network(self.x2)
            with tf.name_scope('similarity'):
                self.similarity = self.predict_similarity(self.output1,self.output2)
                # self.distance = tf.sqrt(tf.reduce_sum(tf.pow(self.output1 - self.output2, 2), axis=1, keep_dims=True))
        
        with tf.name_scope('loss'):
            self.loss = self.contro_loss()
            # self.loss = self.contrastive_loss(self.distance,self.y_true,margin= 0.5)
            # self.loss = self.loss_with_spring()


    def contro_loss(self):
        '''
        s: 网络预测的相似度取值分散在0-1.0之间
        pairs_label:对标签，实际作用是区分类内部分和类间部分分别计算损失，y_true 相当于类内（正对）部分的损失，1-y_true 相当于类间（负对）部分或者说负对部分的损失。
        '''
        
        s = self.predict_similarity(self.output1,self.output2)
        margin = 1.0
        within_part = self.y_true

        # 类内损失：
        # max_part = tf.square(tf.maximum(margin-s,0)) # margin是一个正对该有的相似度临界值，如：1
        differ_loss = margin - s
        #如果相似度s未达到临界值margin，则最小化这个类内损失使s逼近这个margin，增大s
        within_loss = tf.multiply(within_part,differ_loss)
        # 类间损失：
        #如果是负对，between_loss就等于s，这时候within_loss=0，最小化损失就是降低相似度s使之更不相似
        neg_pairs_part = 1.0 - within_part
        between_loss = tf.multiply(neg_pairs_part,s) 

        # 总体损失 = 正对损失+负对损失
        loss = 0.5*(within_loss+between_loss)
        return loss

    def contrastive_loss(self,distance, y, margin):
        with tf.name_scope("contrastive-loss"):
            similarity = y * tf.square(distance)                                           # keep the similar label (1) close to each other
            dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance), 0))        # give penalty to dissimilar label if the distance is bigger than margin
            return 0.5*tf.reduce_mean(dissimilarity + similarity)

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
            h_fc2 = tf.matmul(h_fc1,w_fc2)+b_fc2

        # embedding = h_fc2

        with tf.name_scope('embedding_normalize'):
            embedding = tf.nn.l2_normalize(h_fc2,axis=1)

        return embedding

    def mnist_model_2(self,x):
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

        # with tf.name_scope('fc_x1'):
        #     w_fc_x1 = self.weight_variable([1024,512])
        #     b_fc_x1 = self.bias_variable([512])
        #     h_fc_x1= tf.nn.relu(tf.matmul(h_fc1,w_fc_x1) + b_fc_x1)

        # with tf.name_scope('dropout'):
            # h_fc1_drop = tf.nn.dropout(h_fc1,self.dropout)

        with tf.name_scope('fc2'):
            # embedding in shape: [batch_size,10]
            w_fc2 = self.weight_variable([1024,10])
            self.variable_summaries(w_fc2)
            b_fc2 = self.bias_variable([10])
            self.variable_summaries(b_fc2)
            h_fc2 = tf.matmul(h_fc1,w_fc2)+b_fc2
            # tf.summary.histogram('embedding',embedding)

        with tf.name_scope('embedding_normalize'):
            embedding = tf.nn.l2_normalize(h_fc2,axis=1)
            # tf.reshape(embedding,[1000,10])
        return embedding

    def network(self, x):
        fc1 = self.fc_layer(x, 1024, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, 1024, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.fc_layer(ac2, 2, "fc3")
        return fc3

    def fc_layer(self, inputs_data, n_weight, name):
        assert len(inputs_data.get_shape()) == 2
        n_prev_weight = inputs_data.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(inputs_data, W), b)
        return fc

    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.y_true
        labels_f = tf.subtract(1.0, self.y_true, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.output1, self.output2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def predict_similarity(self,embedding1,embedding2):
        '''
        要注意的是当样本batch_size为1时，用这个当做函数来计算两两样本之间的相似度需要注意最后求均值的axis问题
        '''
        # A, B分别是两个样本经过网络传播之后的提取后的特征/embedding
        # 求两个向量的余弦夹角：A*B/|A|*|B|
        # 求每对样本之间的相似度，即使一个batch_size也是先求各自的再求平均
        cosi = tf.reduce_sum(tf.multiply(embedding1,embedding2),axis=1,keep_dims=True)
        cosi = (cosi+1.0)/2.0 # 平移伸缩变换到[0,1]区间内,谱聚类算法要求的亲和矩阵中不能产生负值。
        # cosi batch_size shape：（batch_size，1）
        return cosi

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