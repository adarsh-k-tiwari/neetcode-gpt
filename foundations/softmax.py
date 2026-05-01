import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_val = np.max(z)
        n = len(z)
        z_sum = 0.0
        for i in range(n):
            z[i] = np.exp(z[i] - max_val)
            z_sum += z[i]
        
        return np.round(z/z_sum, 4)
