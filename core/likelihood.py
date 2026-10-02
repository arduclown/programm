import numpy as np

class Likelihood:
    """
    Вклад одного шага в критерий ММП
    """
    def __init__(self, xp = np):
        """
        :param xp: модуль массивов с интерфейсом numpy
        """
        self._xp = xp

    def value(self, eps: np.ndarray, B: np.ndarray, B_inv: np.ndarray) -> float:
        """
        Значение вклада шага
        :param eps: невязка
        :param B: ковариация невязки
        :param B_inv: обратная матрица к B

        :return: L_k
        """
        _, log_det = self._xp.linalg.slogdet(B)
        #return log_det + float((eps.T @ B_inv @ eps).item())
        return log_det + (eps.T @ B_inv @ eps)[0, 0]
    
    @staticmethod
    def gradient(eps: np.ndarray, B_inv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Производные вклада шага (формулы 28-30)
        :param eps: невязка
        :param B_inv: обратная матрица к ковариации невязки

        :return: (dl/deps, dl/dB)
        """
        dl_deps = 2.0 * B_inv @ eps
        dl_dB = B_inv - B_inv @ eps @ eps.T @ B_inv # формула (30)
        return dl_deps, dl_dB

    @staticmethod
    def derivative(eps: np.ndarray, B_inv: np.ndarray,
                   deps: np.ndarray, dB: np.ndarray) -> np.ndarray:
        """
        Производная вклада шага по каждому параметру (для вычисления 
        градиента при помощи уравнений чувствительности)

        :param eps: невязка
        :param B_inv: обратная к ковариации невязки
        :param deps: производные невязки
        :param dB: производные ковариации невязки
        :return: вектор dl_k/dtheta
        """
        b = B_inv @ eps                      
        return(np.trace(B_inv @ dB)
                + 2.0 * float((deps.T @ b).item())
                - float((b.T @ dB @ b).item()))