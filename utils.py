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


def contro_loss(embedding1,embedding2,pairs_label):

    '''
    总结下来对比损失的特点：首先看标签，然后标签为1是正对，负对部分损失为0，最小化总损失就是最小化类内损失(within_loss)部分，
    让s逼近margin的过程，是个增大的过程；标签为0是负对，正对部分损失为0，最小化总损失就是最小化between_loss，而且此时between_loss就是s，
    所以这个过程也是最小化s的过程，也就使不相似的对更不相似了.
    最小化类内损失的是一个增大s的过程，最小化类间损失的是一个减少s的过程。
    '''
    # cosi = tf.divide(
    #                 tf.reduce_sum(tf.multiply(embedding1,embedding2),axis=1,keep_dims=True),
    #                 tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(embedding1),axis=1,keep_dims=True)),
    #                             tf.sqrt(tf.reduce_sum(tf.square(embedding2),axis=1,keep_dims=True))))

    cosi = tf.reduce_sum(tf.multiply(embedding1,embedding2),axis=1,keep_dims=True)
    s = (cosi+1.0)/2.0 # 平移伸缩变换到[0,1]区间内,谱聚类算法要求的亲和矩阵中不能产生负值。
    # one = tf.constant(1.0)
    margin = 1.0
    diff_part = margin - s

    y_true = pairs_label # y_true = part of within-class part of similarity loss.
    # 类内损失：
    # max_part = tf.square(tf.maximum(margin-s,0)) # margin是一个正对该有的相似度临界值，如：1
    #如果相似度s未达到临界值margin，则最小化这个类内损失使s逼近这个margin，增大s
    within_loss = tf.multiply(y_true,diff_part) #正对相似度要向1靠近以减少损失，从而增大相似度。

    # 类间损失：
    #如果是负对，between_loss就等于s，这时候within_loss=0，最小化损失就是降低相似度s使之更不相似
    between_part = margin - y_true
    between_loss = tf.multiply(between_part,s) # 负对部分的相似度要小

    # 总体损失（要最小化）：
    loss = within_loss+between_loss
    return loss


def predict_similarity(embedding1,embedding2):
    '''
    要注意的是当样本batch_size为1时，用这个当做函数来计算两两样本之间的相似度需要注意最后求均值的axis问题
    cosi shape:(batch_size,1)
    '''
    # A, B分别是两个样本经过网络传播之后的提取后的特征/embedding
    # 求两个向量的余弦夹角：A*B/|A|*|B|
    # 求每对样本之间的相似度，即使一个batch_size也是先求各自的再求平均
    # cosi = tf.divide(
    #                 tf.reduce_sum(tf.multiply(embedding1,embedding2),axis=1,keep_dims=True),
    #                 tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(embedding1),axis=1,keep_dims=True)),
    #                             tf.sqrt(tf.reduce_sum(tf.square(embedding2),axis=1,keep_dims=True))))
    cosi = tf.reduce_sum(tf.multiply(embedding1,embedding2),axis=1,keep_dims=True)
    cosi = (cosi+1.0)/2.0 # 平移伸缩变换到[0,1]区间内,谱聚类算法要求的亲和矩阵中不能产生负值。
    # cosi batch_size 2Dshape：（batch_size，1）
    return cosi

