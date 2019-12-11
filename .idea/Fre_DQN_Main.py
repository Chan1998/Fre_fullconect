import Require_matrixdef as req
import Allocation_matrix as alc
import numpy as np
import random
import tensorflow as tf


n = 1000                    #设定用户数

INPUT_NODE = n * 4
OUTPUT_NODE = n
REGULARIZATION_RATE = 0.0001
LAYER1_NODE = 500
TRAIN_STEPS = 30000

def DNN():
    def inferrence(n, input_tensor, weights1, biases1, weights2, biases2):
        input_tensor = tf.cast(input_tensor, tf.float32)
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        y1 = tf.matmul(layer1, weights2) + biases2
        y2 = tf.sigmoid(y1)
        y2 = tf.round(y2)
        return y2

    x = tf.placeholder(tf.float32, shape=[1, INPUT_NODE], name='x')

    # 生成隐藏层参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    y2 = inferrence(n,x,weights1,biases1,weights2,biases2)

    y_ = tf.placeholder(tf.float32, shape=(OUTPUT_NODE , 1), name='y_')
    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y2,  1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y2, 1e-10, 1.0)))
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy + regularization
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()  # 初始化变量，global初始化所有变量
        sess.run(init_op)
        for k in range(TRAIN_STEPS):
            Req_mat = req.Require_matrix_define(n)
            input_tensor = Req_mat.reshape(1, INPUT_NODE)
            output = sess.run(y2, feed_dict={x: input_tensor})
            output = output.reshape(OUTPUT_NODE ,1)
            Allocated_B_T_matrix, final_result_array, H, success_num = alc.Cost_caculate(n, Req_mat, output)
            y_input = final_result_array
            sess.run(train_step, feed_dict={x: input_tensor,y_ :y_input})

            if k % 1000 == 0:
                print("After %d training steps,H is %d " % (k,H))
        Req_mat = req.Require_matrix_define(n)
        input_tensor = Req_mat.reshape(1, INPUT_NODE)
        output = sess.run(y2, feed_dict={x: input_tensor})
        output = output.reshape(OUTPUT_NODE, 1)
        require_num = n
        result_num = np.sum(output == 1)
        Allocated_B_T_matrix, final_result_array, H, success_num = alc.Cost_caculate(n, Req_mat, output)
        req.Show_Req_B_T_mat(n, Req_mat)
        alc.Show_All_mat(Allocated_B_T_matrix, final_result_array)
        print ("After %d training steps, H  "
               " is %g,require_num = %d,result_number is %d,success_num is %d" % (TRAIN_STEPS, H,require_num , result_num, success_num))


def main():
    DNN()

if __name__ == '__main__':
    main()
#Req_mat = req.Require_matrix_define(n)
#req. Show_Req_B_T_mat(n,Req_mat)
#result_array =  np.random.randint(0,2,size=(n,1))
#Allocated_B_T_matrix,final_result_array,cost,H,success_num=alc.Cost_caculate(n,Req_mat,result_array )
#alc.Show_H_cost_success(H, success_num, cost)
#alc.Show_All_mat(Allocated_B_T_matrix,final_result_array)