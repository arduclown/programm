import numpy as np

def at_step(M, k: int):
    """
    Матрица на шаге k
    :param M: одна матрица (стационарный случай), список матриц по шагам

    :return: матрица на шаге k 
    """
    if M is None:
        return None
    elif isinstance(M, (list, tuple)):
        return M[k]
    elif isinstance(M, np.ndarray) and M.ndim == 3:
        return M[k]
    return M