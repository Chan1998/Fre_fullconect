import Require_matrixdef as req
import Allocation_matrix as alc
import numpy as np
import random



n = 10                    #设定用户数


Req_mat = req.Require_matrix_define(n)
req. Show_Req_B_T_mat(n,Req_mat)
result_array =  np.random.randint(0,2,size=(n,1))
Allocated_B_T_matrix,final_result_array,cost,H,success_num=alc.Cost_caculate(n,Req_mat,result_array )
#alc.Show_H_cost_success(H, success_num, cost)
alc.Show_All_mat(Allocated_B_T_matrix,final_result_array)