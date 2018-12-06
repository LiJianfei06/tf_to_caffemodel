# -*- coding: UTF-8 -*-
import sys
sys.path.append("/home/lijianfei/caffe-master-ljf/python")
sys.path.append("/home/lijianfei/caffe-master-ljf/python/caffe")
import caffe     
from caffe import layers as L,params as P,to_proto
#import tools




root_str="./"



if __name__ == '__main__':
    caffe.set_device(0)  
    caffe.set_mode_gpu() 
#选择 caffe 模型，这里选择第 65000 次迭代的数据
    #snapshot_model_dir = root_str +'model_save/cifar10_ResNet_20_iter_53500.caffemodel'
    snapshot_model_dir = './resnet-20_tf.caffemodel'
    #snapshot_model_dir = './resnet-20_tf_BGR.caffemodel'
#我们只关注测试阶段的结果，因此只写入 test.prototxt
    #test_prototxt_dir = "./test_ResNet_20.prototxt" 
    test_prototxt_dir = "./test_ResNet_20_tf.prototxt" 
    net = caffe.Net(str(test_prototxt_dir), str(snapshot_model_dir), caffe.TEST)
    sum = 0
#测试 1000 次，取平均值
    for _ in range(100):
        net.forward()
        sum += net.blobs['prob'].data
        print net.blobs['prob'].data
    sum /= 100 
    print "sum:",sum
