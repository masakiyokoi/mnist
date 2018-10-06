##mnistのロード

import os
import numpy as np
import struct


def load_mnist(path,kind):
    labels_path =  os.path.join(path,'%s-labels-idx1-ubyte' % kind) #train~
    images_path =  os.path.join(path,'%s-images-idx3-ubyte' % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic,n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype = np.uint8)
            
    with open(images_path, 'rb') as imgpath:
        magic,num,rows,cols = struct.unpack(">IIII",imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels),784)
    
    return images,labels
 