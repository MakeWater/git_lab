# -*- coding: utf-8 -*-
import random
import numpy as np
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


    # wait_to_compare_list = random.sample(np.arange(len(index_to_val)),10) # each row as cluster's label
    # NMI_score = []
    # for row in wait_to_compare_list:
    #     nmi_score = nmi(label[index_to_val[row]], [row for _ in range(len(index_to_val[row]))])
    #     NMI_score.append(nmi_score)
    # NMI_mean_score = np.mean(NMI_score)