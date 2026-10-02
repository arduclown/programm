import numpy as np

from core.model import ModelBase, ModelMatrices, MatrixGradient
from filters.base import FilterBase, FilterRun

class BackwardGradient:
    """
    Вычисление градиента критерия обратным методом:
    Сначала запуск ФК с сохранением истории, затем 
    обратный проход для вычисления градиента функции 
    правдоподобия.

    Property
    ---------------
    filter
        Используемая модификация фильтра Калмана
    """
    name = "обратный"

    def __init__(self, filter: FilterBase):
        self._filter = filter

    @property
    def filter(self) -> FilterBase:
        return self._filter
    
    def value_and_gradient(self, model: ModelBase, theta: np.ndarray, y: list,
                            u: list = None) -> tuple[float, np.ndarray]:
        """
        Значения критерия ММП и его градиент по параметрам
        :param model: модель
        :param theta: вектор неизвестных параметров
        :param y: вектор измерений
        :param u: управляющее воздействие

        :return: значение криетрия L и его производные по тетта
        """
        matrices = model.matrices(theta)
        run = self._filter.run(y, matrices, u)
        grads = self.matrix_gradients(run, matrices)
        return run.loss, grads.to_parameters(model.derivatives(theta))

    def matrix_gradients(self, run: FilterRun, matrices: ModelMatrices) -> MatrixGradients:
        """
        Вычисление градиента 
        :param run: результаты "прямого прохода" фильтра
        :param matrices: матрицы модели
        
        :return: градиенты по матрицам с неизвестными параметрыми и
                    по измерениям
        """
        grads = MatrixGradient(matrices, run.steps)
        grad_state = self._filter.zero_state_gradient(matrices)

        for k in reversed(range(run.steps)):
            grad_pred = self._filter.correct_backward(
                grad_state, run.correct_records[k], matrices, k, grads)
            grad_state = self._filter.predict_backward(
                grad_pred, run.predict_records[k], matrices, k, grads)
        
        self._filter.inital_state_backward(grad_state, matrices, grads)
        
        return grads