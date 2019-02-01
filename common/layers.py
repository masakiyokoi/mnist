# coding: utf-8
import numpy as np
from functions import *


class Affine:
    def __init__(self,W,b):
        
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self,x):#順伝搬
        #print('Affine',x.shape,self.W.shape)
       
        self.x = x
        out = np.dot(x,self.W) + self.b
        
        return out #Affineの出力
    
    def backward(self,dout): #逆伝搬
        
        dx = np.dot(dout,self.W.T) #X側に流れる
        self.dW = np.dot(self.x.T,dout) #W側に流れる
        self.db = np.sum(dout,axis = 0) #バイアス側に流れる
        
        return dx
        
       
    
class Relu:
    
    def __init__(self):
        self.mask = None
        
    def forward(self,x):
        #print('Relu',x.shape)
        #print(x)
        self.mask = (x <= 0)
        #print('Relu',self.mask)
        out = x.copy()
        out[self.mask] = 0 #Relu関数の性質:x<=0はTrue,他はFalseの行列out
        
        return out
    
    def backward(self,dout):
        dout[self.mask] = 0 #<=0の時は何も返さない。
        dx = dout
        
        return dx
        


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None #損失
        self.y = None #softmaxの出力
        self.t = None #正解データ(one-hot)
        
    def forward(self,x,t):
        #print('Softmaxwithloss',x.shape)
        self.t = t
        self.y =softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)
        
        return self.loss
    
    def backward(self,dout = 1):
        batch_size = self.t.shape[0] #正解ラベルの行数
        dx = (self.y - self.t) / batch_size #データ1個あたりの誤差を前のレイヤに送る。
        
        return dx
        
        

    
    
    
