#coding:utf-8
import sys
import os
import copy
import tensorflow as tf
import numpy as np
os.environ['GLOG_minloglevel'] = '2'
sys.path.append("/home/lijianfei/caffe-yolov2/python")
import caffe
import ResNet

from tensorflow.python import pywrap_tensorflow

slim = tf.contrib.slim # 使用方便的contrib.slim库来辅助创建ResNet



cf_prototxt = "./test_ResNet_20_tf.prototxt"
cf_ckpt = "./models_resnet20/model.ckpt-59500"
cf_model = "./resnet-20_tf.caffemodel"


caffe.set_device(0)
caffe.set_mode_gpu()



def print_caffe_params_name():
    net = caffe.Net(cf_prototxt, caffe.TEST)
    print "caffe params:"
    for param_name in net.params.keys():
        for n in range(len(net.params[param_name])):
            print "param_name:",param_name,"shape:",net.params[param_name][n].data.shape



def print_tf_params_name():
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session()
    saver = tf.train.import_meta_graph(cf_ckpt+'.meta') #导入训练数据流图
    saver.restore(sess, cf_ckpt)
    tf_all_vars = tf.global_variables()
    #tf_all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for v in tf_all_vars:
        name = v.name
        print "name:",name

    sess.close()


def tf_ckpt_to_caffemodel(RGB_to_BGR=False):
    i=0
    net = caffe.Net(cf_prototxt, caffe.TEST)        # 加载caffe 的 prototxt
    #------------------------------------------- 按需求增长显存       
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session()

    saver = tf.train.import_meta_graph(cf_ckpt+'.meta') #导入训练数据流图
    saver.restore(sess, cf_ckpt)
    tf_all_vars = tf.global_variables()
    #tf_all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for v in tf_all_vars:
        name = v.name
        name=name.split(":")[0]
        layer_type=name.split("/")[-1]
        #print "layer_type:",layer_type
        if((layer_type=="biases") or (layer_type=="beta") or 
                (layer_type=="moving_variance") or (layer_type=="weights") or 
                (layer_type=="gamma") or (layer_type=="moving_mean")):
            i+=1
            v_4d = sess.run(v) #get the real parameters
            if v_4d.ndim == 4:
                #tf: [ H, W, I, O ] caffe:[ O I H W ]
                v_4d=v_4d.transpose((3, 2, 0, 1))
                #v_4d = np.swapaxes(v_4d, 0, 3) # swap  O W I H 
                #v_4d = np.swapaxes(v_4d, 1, 2) # swap  O I W H
                #v_4d = np.swapaxes(v_4d, 2, 3) # swap  O I H W
            elif v_4d.ndim == 2: #fc
                v_4d=v_4d.transpose((1, 0))

            print i,": ",name,":  v_4d.shape:",v_4d.shape
            if((layer_type=="weights") or (layer_type=="gamma") or (layer_type=="moving_mean")):
                last_name=name
                if((1==i) and (RGB_to_BGR==True) and (layer_type=="weights")):
                    net.params[name][0].data[:,0,:,:]=np.array(v_4d)[:,2,:,:].copy()
                    net.params[name][0].data[:,1,:,:]=np.array(v_4d)[:,1,:,:].copy()
                    net.params[name][0].data[:,2,:,:]=np.array(v_4d)[:,0,:,:].copy()
                    print "RGB_to_BGR!"
                else:
                    net.params[name][0].data[...]=np.array(v_4d)[...].copy()

            elif((layer_type=="biases") or (layer_type=="beta") or (layer_type=="moving_variance")):
                net.params[last_name][1].data[...]=np.array(v_4d)[...].copy()
                if layer_type=="moving_variance":
                    net.params[last_name][2].data[...]=1.0

    sess.close()
    net.save(cf_model)


if __name__=="__main__":
    #print_caffe_params_name()
    #print_tf_params_name()
    tf_ckpt_to_caffemodel(RGB_to_BGR=False)



