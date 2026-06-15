import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        m = np.max(z)
        den = 0
        for i in z:
            den += np.exp(i - m)
        res = []
        for i in z:
            res.append(np.exp(i - m) / den)
        return np.round(np.array(res), 4)

