import Require_matrixdef as req
import Allocation_matrix as alc




n = 10                    #设定用户数


req.Require_matrix_define(n)
req.Show_Req_mat()
result_array =  np.random.randint(0,2,size=(n,1))
Req_mat = req.Require_matrix_define(n)
alc.Cost_caculate(n,Req_mat,result_array  )
alc.Show_All_mat()