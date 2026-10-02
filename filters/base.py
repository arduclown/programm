from abc import ABC, abstractmethod

import numpy as np

from core.model import ModelMatrices, MatrixGradient
from core.matrix_utils import at_step

class FilterRun:
    """
    Результаты работы фильтра.

    Atributes
    ---------------
    loss
        Значение критерия L
    predict_records, correct_records
        Значения, сохранённые на каждом шаге работы фильтра
    
    Property
    ---------------
    steps
        Число шагов
    """

    def __init__(self):
        self.loss = 0.0
        self.predict_records = []
        self.correct_records = []

    @property
    def steps(self) -> int:
        return len(self.correct_records)

class FilterBase(ABC):
    """
    Шаблон модификации фильтра Калмана.

    Каждая модификация описывает только один шаг (прогноз + коррекция)
    и производные этого шага. 

    Состоянием фильтра будем считать то, что переносится с шага на шаг.
    """
    def run(self, y: list, matrices: ModelMatrices, u: list = None) -> FilterRun:
        """
        Запуск фильтра с сохранением всей информации, которая нужна для
        обратного вычисления градиента.

        :param y: измерения
        :param matrices: матрицы модели
        :param u: управляющие воздействие

        :return: значение критерия ММП и сохранённые величины
        """
        result = FilterRun()
        state = self.initial_state(matrices)
        for k in range(len(y)):
            state, predict_record = self.predict(state, matrices, k, at_step(u, k))
            state, loss_k, correct_records = self.correct(state, y[k], matrices, k)
            result.loss += loss_k
            result.predict_records.append(predict_record)
            result.correct_records.append(correct_records)
        return result

    def initial_state(self, matrices: ModelMatrices):
        """
        Начальное состояние фильтра
        :param matrices: матрицы модели

        :return: (x0, P0)
        """
        return matrices.x0, matrices.P0
    
    def zero_state_gradient(self, matrices: ModelMatrices):
        """
        Нулевой градиент по состоянию
        :param matrices: матрицы модели
        :return: (dL/dx, dL/dP) = (0, 0)
        """
        n = matrices.state_dim
        return np.zeros((n, 1)), np.zeros((n, n))

    def inital_state_backward(self, grad_state, matrices: ModelMatrices,
                            grads: MatrixGradient) -> None:
        """
        Перенос градиента по начальному состоянию в градиент по x0, P0
        :param grad_state: градиент по начальному состоянию
        :param matrices: матрицы модели
        :param grads: накапливание градиента
        """
        grads.x0, grads.P0 = grad_state

    # ---- ПРЯМОЙ ШАГ (ФИЛЬТРАЦИЯ ----
    @abstractmethod
    def predict(self, state, matrices: ModelMatrices, k: int, u_k: np.ndarray = None):
        """
        Этап предсказания
        :param state: апостериорное состояние с предыдущего шага
        :param matrices: матрицы модели
        :param k: номер шага
        :param u_k: управляюще воздействие
        
        :return: (прогноз состояния, сохранённые величины)
        """

    @abstractmethod
    def correct(self, predicted_state, y_k: np.ndarray, matrices: ModelMatrices, k: int):
        """
        Этап коррекции (фильтрации)
        :param predicted_step: пргноз состояния
        :param y_k: измерение
        :param matrices: матрицы модели
        :param k: номер шага
        
        :return: (апостериорное состояние, вклад шага в критерий ММП, 
                    сохранённые величины)
        """

    # ---- ОБРАТНЫЙ ШАГ (ВЫЧИСЛЕНИЕ ГРАДИЕНТА) ----
    @abstractmethod
    def correct_backward(self, grad_after, record, matrices: ModelMatrices, k: int,
                        grads: MatrixGradient):
        """
        Обратный проход через коррекцию
        :param grad_after: dL/d(апостериорное сосотяние) от последующих шагов
        :param record: величины, записанные в correct
        :param matrices: матрицы модели
        :param k: номер шага
        :param grads: накапливание градиента по матрицам

        :return: dL/d(прогнозное состояние)
        """

    @abstractmethod
    def predict_backward(self, grad_after, record, matrices: ModelMatrices, k: int,
                        grads: MatrixGradient):
        """
        Обратный проход через прогноз
        :param grad_after: dL/d(прогнозное сосотяние)
        :param record: величины, записанные в predict
        :param matrices: матрицы модели
        :param k: номер шага
        :param grads: накапливание градиента по матрицам

        :return: dL/d(апостореорное состояние с предыдущего шаг)
        """
    
    # ---- СТАНДАРТНОЕ ВЫЧСИЛЕНИЕ ГРАДИЕНТА ----
    def initial_sens(self, matrices: ModelMatrices, derivs: dict):
        """
        Производные начального состояния по неизвестным параметрам

        :param matrices: матрицы модели
        :param derives: производные матриу модели по параметрам

        :return: (dx0/dTheta, dP0/dTheta)
        """
        return derivs["x0"].copy(), derivs["P0"].copy()
    
    def predict_forward(self, sens, state, matrices: ModelMatrices,
                        derivs: dict, k: int):
        """
        Прямое вычисление производных через прогноз состояния

        :param sens: производные апостериорного состояния по тетта (с предыдущего шага)
        :param state: само апостериорное состояние с предыдущего шага
        :param matrices: матрицы модели
        :param derivs: производные матриц модели по параметрам
        :param k: номер шага

        :return: производные прогнозного состояния по тетта
        """
    
    def correct_forward(sens, sens_pred, predictes_state, record, matrices: ModelMatrices,
                        derivs: dict, k: int):
        """
        Прямой вычисление производных через коррекцию

        :param sens_pred: производные прогнозного состояния по тетта
        :param predicted_state: само прогнозное состояние
        :param record: величины, сохранённые в correct на этом шаге
        :param matrices: матрицы модели
        :param derivs: производные матриц модели по параметрам
        :param k: номер шага

        :return: (производные апостериорного состояния по тетта, вектор dl_k/dThetta)
        """
