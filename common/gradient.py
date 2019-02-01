# coding: utf-8
import numpy as np

def numerical_gradient(f,x):#関数fに対してパラメータの更新
    h = 1e-4 #0.0001
    grad = np.zeros_like(x) #入力xと同じ形で要素が0の配列作成
    it = np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index #idxに行列xの各要素のインデックスを入れる。(0,0),(0,1),,,,,,,,
        tmp_val = x[idx] #行列xのある一つの要素の値がtmp_valに入っていく。
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) #f(x+h):厳密には一つの要素が変わっただけ.whileの最後には全ての要素が+hされる。
        
        x[idx] = float(tmp_val) - h
        fxh2 = f(x) #f(x-h)
        grad[idx] = (fxh1 - fxh2) /(2*h) #gradの一つの要素に数値微分の結果を代入
        
        x[idx] = tmp_val #数値をもとに戻す。
        it.iternext()
    return grad
        
        
    
    
