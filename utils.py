# -*- coding: utf-8 -*-
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import normalized_mutual_info_score as nmi

def NMI(label_pred,label):
    '''convenient function to evaluate model,in NMI score;
    index_to_val: the array contains each subcluster. use it's row number as cluster label
    label: the True label of each subcluster in index_to_val,the row in index_to_val was consist by index of raw data
    return: NMI score that show the purity of subcluster.
    '''
    # used in run.py
    NMI_score = nmi(label_pred,label)
    print('the mean NMI score is:',NMI_score)
    return NMI_score

def batch_generator(pairs,pairs_label,batch_size):
    # 这只是一个生成器函数，调用这个函数只是返回一个生成器对象，而不是数据哦！！
    # 可以在for循环中调用生成器对象。
    assert pairs.shape[0] == pairs_label.shape[0],('pairs.shape:%s labels.shape:%s' % (pairs.shape,labels.shape))
    num_examples = len(pairs) # 样本数量
    batch_num = int(len(pairs)/batch_size) + 1 # 在训练集上遍历一遍（一轮）需要的batch数量
    for num in range(batch_num):
        start_index = num * batch_size
        end_index = min((num+1)*batch_size,num_examples)
        # random_indx = np.random.permutation(len(pairs))  # 具有打乱数据功能的
        # permut = random_indx[start_index:end_index]
        # 如果数据输入前已经打乱了，就没必要再去打乱了
        # 多维数组切片，先取出第一维的（即哪些pairs、pairs_labels)，再选择其他维的(比如每一维的第一维或者第二维等等)。
        batch_slice = np.arange(start_index,end_index) 
        x1, x2 = pairs[batch_slice,0],pairs[batch_slice,1]
        y = pairs_label[batch_slice]
        yield ([x1,x2],y)

def deepnn(x):

    # input reshape to [batch_size,28,28,channel]
    with tf.name_scope('reshape'):
        # transform input type to tensorflow type
        # x = tf.cast(x,tf.float32)
        x_image = tf.reshape(x,[-1,28,28,1])
        tf.summary.image('input',x_image,10)

    # layer1: picture width = 28->28
    with tf.name_scope('conv1'):
        w_conv1 = weight_variable([5,5,1,32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
        tf.summary.histogram('activation1',h_conv1)

    # pooling layer1 : 28->14
    with tf.name_scope('pooling1'):
        h_pooling_1 = max_pool_2x2(h_conv1)

    # convolution layer2 : 14->14
    with tf.name_scope('conv2'):
        w_conv2 = weight_variable([5,5,32,64])
        b_conv2 = weight_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pooling_1,w_conv2) + b_conv2)
        tf.summary.histogram('activation2',h_conv2)

    # feature width : 14->7
    with tf.name_scope('pooling2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
        # W2 = (W1-F+2P)/S + 1
        w_fc1 = weight_variable([7*7*64,1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)
        tf.summary.histogram('activation3',h_fc1)

    # with tf.name_scope('dropout'):
        # h_fc1_drop = tf.nn.dropout(h_fc1,dropout)

    with tf.name_scope('fc2'):
        # embedding in shape: [batch_size,10]
        w_fc2 = weight_variable([1024,10])
        b_fc2 = bias_variable([10])
        embedding = tf.matmul(h_fc1,w_fc2)+b_fc2
        tf.summary.histogram('embedding',embedding)
        # embedding = tf.nn.softmax(embedding,dim=1)
        # tf.reshape(embedding,[1000,10])
    return embedding


def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

def contro_loss(similarity_by_network,pairs_label):

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
    max_part = tf.square(tf.maximum(margin-s,0)) # margin是一个正对该有的相似度临界值，如：1
    #如果相似度s未达到临界值margin，则最小化这个类内损失使s逼近这个margin，增大s
    within_loss = tf.multiply(y_true,max_part) 

    # 类间损失：
    #如果是负对，between_loss就等于s，这时候within_loss=0，最小化损失就是降低相似度s使之更不相似
    between_loss = tf.multiply(1.0-y_true,s) 

    # 总体损失（要最小化）：
    loss = 0.5*tf.reduce_mean(within_loss+between_loss) 
    return loss

def predict_similarity(embedding1,embedding2):
    '''
    要注意的是当样本batch_size为1时，用这个当做函数来计算两两样本之间的相似度需要注意最后求均值的axis问题
    '''
    # A, B分别是两个样本经过网络传播之后的提取后的特征/embedding
    # 求两个向量的余弦夹角：A*B/|A|*|B|
    # 求每对样本之间的相似度，即使一个batch_size也是先求各自的再求平均
    cosi = tf.reduce_mean(tf.divide(tf.reduce_sum(tf.multiply(embedding1,embedding2),axis=1,keep_dims=True),
                tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(embedding1),axis=1)),
                tf.sqrt(tf.reduce_sum(tf.square(embedding2),axis=1,keep_dims=True)))))
    cosi = (cosi+1)/2.0 # 平移伸缩变换到[0,1]区间内,谱聚类算法要求的亲和矩阵中不能产生负值。
    # cosi batch_size shape：（batch_size，1）
    return cosi