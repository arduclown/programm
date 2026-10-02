import numpy as np

from core.model import ModelBase, complete_derivatives
from filters.base import FilterBase
from core.matrix_utils import at_step

class ForwardSensivityGradient:
    """
    Градиент критерия ММП с использованием уравнений чувствительности: 

    Property
    ---------------
    filter
        Используемая модификация фильтра Калмана
    """
    name = "прямой"

    def __init__(self, filter: FilterBase):
        self._filter = filter

    @property
    def filter(self) -> FilterBase:
        return self._filter
    
    def value_and_gradient(self, model: ModelBase, theta: np.ndarray, y: list,
                            u: list = None) -> tuple[float, np.ndarray]:
        """
        Значение критерия ММП и его градиента по параметрам

        :param model: модель
        :param theta: вектор неизвестных параметров
        :param y: вектор измерений
        :param u: управляющее воздействие

        :return: значение криетрия L и его производные по тетта
        """
        matrices = model.matrices(theta)
        derivs = complete_derivatives(matrices, model.derivatives(theta))
        kf = self._filter

        state = kf.initial_state(matrices)
        sens = [kf.initial_sens(matrices, d) for d in derivs]
        loss = 0.0
        grad = np.zeros(len(derivs))

        for k in range(len(y)):
            predict_state, _ = kf.predict(state, matrices, k, at_step(u, k))
            sens_pred = [kf.predict_forward(sens[i], state, matrices, d, k)
                        for i, d in enumerate(derivs)]

            state, loss_k, record = kf.correct(predict_state, y[k], matrices, k)
            loss += loss_k
            for i, d in enumerate(derivs):
                sens[i], dloss_k = kf.correct_forward(sens_pred[i], predict_state, record,
                                                        matrices, d, k)
                grad[i] += dloss_k

        return loss, grad
        

