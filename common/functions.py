# coding: utf-8
import numpy as np

def softmax(z):#活性化関数(zはラベルと同じ形の行列)
    
    c = np.max(z) #ラベル(1×10)の中で一番大きい値
    exp_z = np.exp(z - c) #全ての要素を最大値から減算し
    sum_exp_z =np.sum(exp_z)
    y = exp_z / sum_exp_z
    
    return y

def cross_entropy_error(y,t):#損失関数
    
    delta = 1e-7
    
    return -np.sum(t * np.log(y + delta)) #logの中身が0にならないようdelta入れる


    
    
    