import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import Require_matrixdef as req

#S_N=20                          #这里我们假设通信信噪比固定为20dB


def cost_caculate(n,Req_mat,result_array):
    Allocated_B_T_matrix = np.zeros((BAND_WIDTH, TIME_LENGTH), dtype=int)
    H = 0
    success_num = 0
    cost = 0
    final_result_array = np.zeros((n,1),dtype=int)
    for i in range (n):
        if result_array[i,0]:
            if np.all(Allocated_B_T_matrix[int(round(c_b[i,0]-0.5*b[i,0])):int( round(c_b[i,0] + 0.5*b[i,0])),
                      int(round(c_t[i,0]-0.5*t[i,0])): int(round(c_t[i,0] + 0.5*t[i,0]))] == 0):
                H = H + b[i,0] * 1000 * t[i,0] * 60 * 6.65              #这里对数直接计算了，不再倒入math函数了香农定理  R= b*log（1+ s/n）
                success_num = success_num +1
                Allocated_B_T_matrix[int(round(c_b[i, 0] - 0.5 * b[i, 0])):int(round(c_b[i, 0] + 0.5 * b[i, 0])),
                int(round(c_t[i, 0] - 0.5 * t[i, 0])): int(round(c_t[i, 0] + 0.5 * t[i, 0]))] = 1
                final_result_array[i,0] = 1
    success_rate = success_num / n
    cost = -(H + success_rate)
    print ("H is %d,success_rate is %d,cost is %d" % (H, success_rate, cost))

    sns.heatmap(Allocated_B_T_matrix, annot=False, vmin=0, vmax=3, center=1, cmap="Blues", xticklabels=False,
                yticklabels=False)
    plt.xlabel('Time')
    plt.ylabel('Frequence')
    plt.show()

    sns.heatmap(final_result_array , annot=False, vmin=0, vmax=3, center=1, cmap="Blues", xticklabels=False,
                yticklabels=False)
    plt.xlabel('decision')
    plt.ylabel('applicant')
    plt.show()
    print (final_result_array)

    return cost


#n = 10
#result_array =  np.random.randint(0,2,size=(n,1))
#print (result_array)
#Req_mat = req.Require_matrix_define(n)
#cost_caculate(n,Req_mat,result_array  )
