import numpy as np

from filters.base import FilterBase
from core.model import ModelBase

class FiniteDifferenceGradient:
    """
    Градиент критерия ММП по формуле центральной разности

    Property
    ------------
    kalman_filter
        Используемая модификация фильтра
    eps
        Шаг разности
    """
    name = "разностный"

    def __init__(self, kalman_filter: FilterBase, eps: float = 1e-6):
        self._filter = kalman_filter
        self._eps = eps

    @property
    def kalman_filter(self) -> FilterBase:
        return self._filter
    
    @property
    def eps(self) -> float:
        return self._eps
    
    def loss(self, model: ModelBase, theta: np.ndarray, y: list, u: list = None) -> float:
        """
        Значение критерия

        :param model: модель
        :param theta: вектор параметров
        :param y: измерения
        :param u: управляющее воздействие
        
        :return: L
        """
        return self._filter.run(y, model.matrices(theta), u).loss
    
    def value_and_gradient(self, model: ModelBase, theta: np.ndarray, y: list,
                            u: list = None) -> tuple[float, np.ndarray]:
        """
        Значение критерия и его разностный градиент по параметрам

        :param model: модель
        :param theta: вектор параметров
        :param y: измерения
        :param u: управляющее воздействие

        :return: (L, dL/dTheta)
        """
        grad = np.zeros(len(theta))
        for i in range(len(theta)):
            theta_plus, theta_minus = theta.copy(), theta.copy()
            theta_plus[i] += self._eps
            theta_minus[i] -= self._eps
            grad[i] = (self.loss(model, theta_plus, y, u)
                        - self.loss(model, theta_minus, y, u)) / (2 * self._eps)
        return self.loss(model, theta, y, u), grad