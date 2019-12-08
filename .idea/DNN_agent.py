import tensorflow as tf
import numpy as np
import Require_matrixdef as req
import Allocation_matrix as alc

INPUT_NODE = n * 4          #输入层节点数
OUTPUT_NODE = n             #输出层节点数

#配置神经网络参数
Req_mat_num = 300000        #生成申请数量
LAYER1_NODE = 500   #隐藏层节点数。此处使用有500个节点的一个隐藏层的网络结构
BATCH_SIZE = 100    #一个训练batch中训练数据个数
LEARNING_RATE_BASE = 0.8        #基础学习率
LEARNING_RATE_DECAY = 0.99      #学习率的衰减率
REGULARIZATION_RATE = 0.0001    #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 3000         #训练轮数
MOVING_AVERAGE_DECAY = 0.99     #滑动平均衰减率

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    #当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        #计算隐藏层前向传播结果，使用ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) +biases2

    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) +avg_class.average(biases2)


#训练模型
def train(Fre_allca_op,cost):
    x = tf.placeholder(tf.int, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.int, [None, OUTPUT_NODE], name='y-input')

    #生成隐藏层参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    #生成输出层参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    y = inference(x, None, weights1, biases1, weights2, biases2)
    global_step = tf.Variable(0, trainable = False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)

    loss = cost + regularization

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, Req_mat_num/BATCH_SIZE, LEARNING_RATE_DECAY)


    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)


    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')


    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
