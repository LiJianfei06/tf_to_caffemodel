# coding:UTF-8
"""

Typical use:

   from tensorflow.contrib.slim.nets import resnet_v2

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope(is_training)):
      net, end_points = resnet_v2.resnet_v2_101(inputs, 1000)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope(is_training)):
      net, end_points = resnet_v2.resnet_v2_101(inputs,
                                                21,
                                                global_pool=False,
                                                output_stride=16)
"""
import collections # 原生的collections库
import tensorflow as tf
import numpy as np
import sys
from datetime import datetime
import math
import time
from tensorflow.python.ops import init_ops



slim = tf.contrib.slim # 使用方便的contrib.slim库来辅助创建ResNet

FLAGS = tf.app.flags.FLAGS


# 图像大小  
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32 

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  '''
  使用collections.namedtuple设计ResNet基本模块组的name tuple，并用它创建Block的类
  只包含数据结构，不包含具体方法。
  定义一个典型的Block，需要输入三个参数：
  scope：Block的名称
  unit_fn：ResNet V2中的残差学习单元 
  args：Block的args。
  '''


"""
  shortcut
  是否需要下采样
"""
def subsample(inputs, stride, num_outputs,is_training=True,keep_prob=1.0,scope=None): 
  if stride == 1:       # 平常的shortcut
    return inputs
  else:                 # 下采样时的shortcut 用1x1卷积
    #return slim.max_pool2d(inputs, [1, 1], stride=stride, scope=scope)
      shortcut=slim.conv2d(inputs, num_outputs, 1, stride=stride,
                       #normalizer_fn=None, 
                       activation_fn=None,
                       padding='VALID', scope=scope)
      #shortcut = slim.dropout(shortcut, keep_prob)
      #return slim.batch_norm(shortcut, is_training=is_training,activation_fn=tf.nn.relu) 
      return shortcut



"""
创建一个卷积层  
"""
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None): 
  #return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,padding='SAME', scope=scope)
  if stride == 1:
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                       padding='SAME', scope=scope)
  else:                             # 如果不为1，则显式的pad zero，pad zero总数为kernel_size - 1
    #kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs, # 对输入变量进行补零操作
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                       #normalizer_fn=None, activation_fn=None,
                       padding='VALID', scope=scope)


"""
定义堆叠Blocks的函数 
"""
@slim.add_arg_scope
def stack_blocks_dense(net, blocks,
                       outputs_collections=None,is_training=True,keep_prob=1.0):
  """
  Args:
    net: A `Tensor` of size [batch, height, width, channels].输入。
    blocks: 是之前定义的Block的class的列表。
    outputs_collections: 收集各个end_points的collections。

  Returns:
    net: Output tensor 

  """
  # 使用两层循环，逐个Residual Unit地堆叠
  for block in blocks: # 先使用两个tf.variable_scope将残差学习单元命名为block1/unit_1的形式
    #print "block:",block
    with tf.variable_scope(block.scope, 'block', [net]) as sc:
      for i, unit in enumerate(block.args):
        #print "i:",i
        #print "unit:",unit
        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
          # 在第2层循环中，我们拿到每个block中每个Residual Unit的args并展开为下面四个参数
          unit_depth, unit_depth_bottleneck, unit_stride = unit
          print  unit_depth, unit_depth_bottleneck, unit_stride
          net = block.unit_fn(net, # 使用残差学习单元的生成函数顺序的创建并连接所有的残差学习单元
                              depth=unit_depth,
                              depth_bottleneck=unit_depth_bottleneck,
                              stride=unit_stride,is_training=is_training,keep_prob=keep_prob)
          #print "net:",net
      net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net) # 将输出net添加到collections中

  return net # 当所有block中的所有Residual Unit都堆叠完成之后，再返回最后的net作为stack_blocks_dense





"""
创建ResNet通用的arg_scope,arg_scope用来定义某些函数的参数默认值 
"""
def resnet_arg_scope(is_training=True, # 训练标记
                     weight_decay=5e-4, # 权重衰减速率
                     batch_norm_decay=0.997, # BN的衰减速率
                     batch_norm_epsilon=1e-5, #  BN的epsilon默认1e-5
                     batch_norm_scale=True): # BN的scale默认值

  batch_norm_params = { # 定义batch normalization（标准化）的参数字典
      'is_training': is_training,
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }

  with slim.arg_scope( # 通过slim.arg_scope将[slim.conv2d]的几个默认参数设置好
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay), # 权重正则器设置为L2正则 
      weights_initializer=slim.xavier_initializer(), # 权重初始化器
      biases_initializer=init_ops.zeros_initializer(),
      biases_regularizer=None,
      activation_fn=tf.nn.relu, # 激活函数
      normalizer_fn=slim.batch_norm, # 标准化器设置为BN
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc: # ResNet原论文是VALID模式，SAME模式可让特征对齐更简单
        return arg_sc # 最后将基层嵌套的arg_scope作为结果返回


@slim.add_arg_scope
def bottleneck1(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None,is_training=True,keep_prob=1.0):
  """
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth、depth_bottleneck:、stride三个参数是前面blocks类中的args
    rate: An integer, rate for atrous convolution.
    outputs_collections: 是收集end_points的collection
    scope: 是这个unit的名称。
  """
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc: 
    #depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4) # 可以限定最少为四个维度

    shortcut = subsample(inputs, stride,depth,is_training,keep_prob, 'shortcut')

    # 先是一个1*1尺寸，步长1，输出通道数为depth_bottleneck的卷积
    # 然后是3*3尺寸，步长为stride，输出通道数为depth_bottleneck的卷积
    if stride==1:
        residual = slim.conv2d(inputs, depth_bottleneck, 3, stride=stride,padding='SAME', scope='conv1')
    elif stride==2:
        inputs = tf.pad(inputs, [[0, 0], [1, 0], [1, 0], [0, 0]])
        residual = slim.conv2d(inputs, depth_bottleneck, 3, stride=stride,padding='VALID', scope='conv1')

    print "residual:",residual
    tf.summary.histogram('conv1', residual)

    # 最后是1*1卷积，步长1，输出通道数depth的卷积，得到最终的residual。最后一层没有正则项也没有激活函数
    residual = slim.conv2d(residual, depth, 3, stride=1,activation_fn=None,padding='SAME', scope='conv2')
    print "residual:",residual
    tf.summary.histogram('conv2', residual)

    output_temp = shortcut + residual # 将降采样的结果和residual相加
    output = tf.nn.relu(output_temp)

    return slim.utils.collect_named_outputs(outputs_collections, # 将output添加进collection并返回output作为函数结果
                                            sc.name,
                                            output)
    


########定义生成resnet_v2的主函数########
def resnet_v2(inputs, # A tensor of size [batch, height_in, width_in, channels].输入
              blocks, # 定义好的Block类的列表
              num_classes=None, # 最后输出的类数
              global_pool=True, # 是否加上最后的一层全局平均池化
              include_root_block=True, # 是否加上ResNet网络最前面通常使用的卷积和最大池化
              reuse=None, # 是否重用
              scope=None,# 整个网络的名称
              is_training=True,
              keep_prob=1.0): 
  # 在函数体先定义好variable_scope和end_points_collection
  with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + 'end_points' # 定义end_points_collection
    #sys.exit()
    with slim.arg_scope([slim.conv2d, bottleneck1,          # bottleneck1bottleneck1bottleneck1bottleneck1
                         stack_blocks_dense],
                        outputs_collections=end_points_collection): # 将三个参数的outputs_collections默认设置为end_points_collection

      net = inputs
      #print "net:",net      
      if include_root_block: # 根据标记值
        with slim.arg_scope([slim.conv2d],
                            #activation_fn=None, normalizer_fn=None
                            ):
          net = slim.conv2d(net, 16, 3, stride=1,padding='SAME', scope='conv1')
          
      net = stack_blocks_dense(net, blocks,is_training=is_training,keep_prob=keep_prob) # 将残差学习模块组生成好

      if global_pool: # 根据标记添加全局平均池化层
        net=slim.avg_pool2d(net, [np.shape(net)[1], np.shape(net)[2]], stride=1, scope='pool5')
        print 'Avg_pool:',net
      if num_classes is not None:  # 是否有通道数
        with slim.arg_scope([slim.conv2d],
                            #activation_fn=None, normalizer_fn=None
                            ):
            net = slim.conv2d(net, num_classes, 1, stride=1, activation_fn=None, # 无激活函数和正则项
                              normalizer_fn=None,
                              scope='logits') # 添加一个输出通道num_classes的1*1的卷积
        #net=slim.fully_connected(net, num_classes,  weights_initializer=slim.xavier_initializer(),
        #                         activation_fn=None, # 无激活函数和正则项
        #                  normalizer_fn=None, scope='logits') # 添加一个输出通道num_classes的1*1的卷积
        net = tf.reshape(net, [-1, num_classes])
      end_points = slim.utils.convert_collection_to_dict(end_points_collection) # 将collection转化为python的dict
      if num_classes is not None:
        end_points['predictions'] = slim.softmax(net, scope='predictions') # 输出网络结果
        print "net:",net
        print "end_points['predictions']:",end_points['predictions']
      return net, end_points
#------------------------------ResNet的生成函数定义好了----------------------------------------
      



"""
  ResNet-20  
  for cifar10
"""
def resnet_20(inputs, # 图像尺寸缩小了32倍
                 num_classes=None,
                 global_pool=True,
                 reuse=None, # 是否重用
                 is_training=True,
                 keep_prob=1.0,
                 scope='resnet_ljf'):

  blocks = [
      Block('block1', bottleneck1, [(16, 16, 1)] * 1 + [(16, 16, 1)]*2),        # (输入通道数,输出通道数,stride)
      Block('block2', bottleneck1, [(32, 32, 2)] * 1 + [(32, 32, 1)]*2),
      Block('block3', bottleneck1, [(64, 64, 2)] * 1 + [(64, 64, 1)]*2),
      ]
  #print "blocks:",blocks
  return resnet_v2(inputs, blocks, num_classes, global_pool,
                   include_root_block=True, reuse=reuse, scope=scope,is_training=is_training,keep_prob=keep_prob)



#def resnet_v2_50(inputs, # 图像尺寸缩小了32倍
#                 num_classes=None,
#                 global_pool=True,
#                 reuse=None, # 是否重用
#                 is_training=True,
#                 scope='resnet_v2_50'):
#  blocks = [
#      Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
#
#
#
#      # Args:：
#      # 'block1'：Block名称（或scope）
#      # bottleneck：ResNet V2残差学习单元
#      # [(256, 64, 1)] * 2 + [(256, 64, 2)]：Block的Args，Args是一个列表。其中每个元素都对应一个bottleneck
#      #                                     前两个元素都是(256, 64, 1)，最后一个是(256, 64, 2）。每个元素
#      #                                     都是一个三元tuple，即（depth，depth_bottleneck，stride）。
#      # (256, 64, 3)代表构建的bottleneck残差学习单元（每个残差学习单元包含三个卷积层）中，第三层输出通道数
#      # depth为256，前两层输出通道数depth_bottleneck为64，且中间那层步长3。这个残差学习单元结构为：
#      # [(1*1/s1,64),(3*3/s2,64),(1*1/s1,256)]
#
#
#
#      Block(
#          'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
#      Block(
#          'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
#      Block(
#          'block4', bottleneck, [(2048, 512, 1)] * 3)]
#  #print "blocks:",blocks
#  return resnet_v2(inputs, blocks, num_classes, global_pool,
#                   include_root_block=True, reuse=reuse, scope=scope,is_training=is_training)
#
#
#def resnet_v2_101(inputs, # unit提升的主要场所是block3
#                  num_classes=None,
#                  global_pool=True,
#                  reuse=None,
#                  scope='resnet_v2_101'):
#  """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
#  blocks = [
#      Block(
#          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
#      Block(
#          'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
#      Block(
#          'block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
#      Block(
#          'block4', bottleneck, [(2048, 512, 1)] * 3)]
#  return resnet_v2(inputs, blocks, num_classes, global_pool,
#                   include_root_block=False, reuse=reuse, scope=scope)
#
#
#def resnet_v2_152(inputs, # unit提升的主要场所是block3
#                  num_classes=None,
#                  global_pool=True,
#                  reuse=None,
#                  scope='resnet_v2_152'):
#  """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
#  blocks = [
#      Block(
#          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
#      Block(
#          'block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
#      Block(
#          'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
#      Block(
#          'block4', bottleneck, [(2048, 512, 1)] * 3)]
#  return resnet_v2(inputs, blocks, num_classes, global_pool,
#                   include_root_block=True, reuse=reuse, scope=scope)
#
#
#def resnet_v2_200(inputs, # unit提升的主要场所是block2
#                  num_classes=None,
#                  global_pool=True,
#                  reuse=None,
#                  scope='resnet_v2_200'):
#  """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
#  blocks = [
#      Block(
#          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
#      Block(
#          'block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
#      Block(
#          'block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
#      Block(
#          'block4', bottleneck, [(2048, 512, 1)] * 3)]
#  return resnet_v2(inputs, blocks, num_classes, global_pool,
#                   include_root_block=True, reuse=reuse, scope=scope)


