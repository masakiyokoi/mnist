import sys,os
sys.path.append('../common')
from functions import softmax,cross_entropy_error  #softmax(活性化関数),エントロピー二乗誤差(誤差関数)
from gradient import numerical_gradient #numerical_gradient(パラメータの更新、勾配)
import pprint
import numpy as np

#重みWはガウス分布で初期化,バイアスは0で初期化
class Twolayer:

    def __init__(self,input_size,hidden_size,output_size,weight_init_std = 0.01):

        #重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size) #学習率(0.01)*ランダムに生成された行列(input_size×hidden_size)
        self.params['b1'] = np.zeros(hidden_size) #(1×hidden_size)、要素が0の1次元配列
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size) #行列の形が違う。
        self.params['b2'] = np.zeros(output_size) #行列の形が違う。

    def convert_t(self,t_train):

        t = np.zeros((t_train.shape[0],10))
        for i in range(t_train.shape[0]):
            label = t_train[i] #5
            t[i][label-1] = 1

        return t


    def predict(self,x): #出力
        W1,W2 = self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']

        a1 = np.dot(x,W1) + b1
        z1 = softmax(a1)
        a2 = np.dot(z1,W2) + b2
        y = softmax(a2)

        return y


    def loss(self,x,t): #lossの算出 by cross_entropy_error
        y = self.predict(x)

        return cross_entropy_error(y,t)

    def accuracy(self,x,t):

        y = self.predict(x)
        y = np.argmax(y,axis = 0)
        t = np.argmax(t,axis = 0)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self,x,t):
        loss_W = lambda W:self.loss(x,t) #入力と正解ラベルのlossを求める無名関数loss_Wの作成

        grads = {}

        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])

        return grads
