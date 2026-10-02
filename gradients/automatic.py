import numpy as np
import torch


from core.model import ModelMatrices, ModelBase
from filters.base import FilterBase

class AutomaticDiff:
    """
    Градиент криетря ММП посчитанный автоматическим диф.
    Property
    ------------
    kalman_filter
        Используемая модификация фильтра Калмана
    """
    name = "autograd"
    MATRIX_NAMES = ("F", "H", "G", "Q", "R", "x0", "P0")

    def __init__(self, kalman_filter: FilterBase):
        """
        :param kalman_filter: фильтр, созданный с xp = torch
        """
        self._filter = kalman_filter

    @property
    def kalman_filter(self) -> FilterBase:
        return self._filter
    
    @staticmethod
    def _to_tensor(value, requires_grad: bool = False):
        """
        Перевод массива в тензоры
        :param value: массив
        :param requires_grad: нужно ли считать град. по этому тензору

        :return: тензор или список тензоров
        """
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [AutomaticDiff._to_tensor(v, requires_grad) for v in value]
        return torch.tensor(np.asarray(value), dtype=torch.float64, requires_grad=requires_grad)
    
    def value_and_gradient(self, model: ModelBase, theta: np.ndarray, y: list,
                            u: list = None) -> tuple[float, np.ndarray]:
        """
        Значение критерия ММП

        :param model: модель
        :param theta: вектор неизвестных параметров
        :param y: вектор измерений
        :param u: управляющее воздействие

        :return: значение криетрия L и его производные по тетта
        """
        matrices = model.matrices(theta)

        leaves = {name: self._to_tensor(getattr(matrices, name), requires_grad=True)
                    for name in self.MATRIX_NAMES}
        torch_matrices = ModelMatrices(**leaves, Psi = self._to_tensor(matrices.Psi))

        loss = self._filter.run(self._to_tensor(y), torch_matrices, self._to_tensor(u)).loss
        loss.backward()

        result = np.zeros(len(theta))
        for i, derivs in enumerate(model.derivatives(theta)):
            for name, dM in derivs.items():
                grad = _leaf_grad(leaves[name])
                result[i] += float(np.tensordot(grad, dM, axes = 2))
        return float(loss.item()), result

def _leaf_grad(leaf) -> np.ndarray:
    """
    Градиент, накопленный на листе
    :param leaf: тензор или список тензоров
    :return: градиент как массив нампай
    """
    if isinstance(leaf, list):
        return sum(_leaf_grad(t) for t in leaf)
    if leaf.grad is None:                       # матрица не участвовала в вычислении
        return np.zeros(tuple(leaf.shape))
    return leaf.grad.numpy()
