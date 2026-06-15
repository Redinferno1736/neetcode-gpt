import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        res=[]
        for num in z:
            res.append(1 / (1 + np.exp(-num)))
        res=np.array(res)
        return np.round(res, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        res=[]
        for num in z:
            res.append(float(max(0,num)))
        res=np.array(res)
        return np.round(res, 5)

