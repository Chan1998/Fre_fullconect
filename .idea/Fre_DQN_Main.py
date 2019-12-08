import Require_matrixdef as req

import Allocation_matrix as all




n = 10                    #设定用户数


req.Require_matrix_define(n)
result_array =  np.random.randint(0,2,size=(n,1))
Req_mat = req.Require_matrix_define(n)
cost_caculate(n,Req_mat,result_array  )
