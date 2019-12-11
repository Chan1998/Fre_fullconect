import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt


BAND_WIDTH = 1000          #带宽1000MHz
TIME_LENGTH = 1440          #时间总长1440分钟
AVG_BAND_WIDTH = 10         #平均使用带宽
AVG_TIME_LONGTH = 1440        #平均使用时长


def Require_matrix_define(n):               #生成申请矩阵
    b = np.zeros((n,1),dtype= int)
    t = np.zeros((n,1),dtype= int)
    c_b = np.zeros((n,1),dtype= int)
    c_t = np.zeros((n,1),dtype= int)
    for i in range (n):
        b[i,0] = round(AVG_BAND_WIDTH * random.random())
        t[i,0] = round(AVG_TIME_LONGTH * random.random())
        c_b[i, 0] = round(BAND_WIDTH * random.random())
        c_t[i, 0] = round(TIME_LENGTH * random.random())
    Require_matrix = np.hstack((b ,t ,c_b ,c_t ))
    return Require_matrix


def Show_Req_B_T_mat(n,Require_matrix):
    # 统计申请矩阵
    Require_B_T_matrix = np.zeros((BAND_WIDTH ,TIME_LENGTH ),dtype=int)
    for i in range (n):
       Require_B_T_matrix[int(round(Require_matrix[i,2]-0.5*Require_matrix[i,0])):int( round(Require_matrix[i,2] + 0.5*Require_matrix[i,0])),
                      int(round(Require_matrix[i,3]-0.5*Require_matrix[i,1])): int(round(Require_matrix[i,3] + 0.5*Require_matrix[i,1]))] \
           = Require_B_T_matrix [int(round(Require_matrix[i,2]-0.5*Require_matrix[i,0])):int( round(Require_matrix[i,2] + 0.5*Require_matrix[i,0])),
                      int(round(Require_matrix[i,3]-0.5*Require_matrix[i,1])): int(round(Require_matrix[i,3] + 0.5*Require_matrix[i,1]))] +1

    sns.heatmap(Require_B_T_matrix ,annot=False , vmin=0, vmax=10, center= 3,cmap= "Blues",xticklabels =False  ,yticklabels =False   )
    plt.xlabel ('Time')
    plt.ylabel ('Frequence')
    plt.show()







#Require_matrix_define(100)
