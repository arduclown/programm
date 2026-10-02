from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from core.matrix_utils import at_step

@dataclass
class ModelMatrices:
    """
    Матрицы линенйной дискретной модели
    
    Property
    ---------------
    state_dim
        Размерность вектора состояния

    """

    F: np.ndarray
    H: np.ndarray
    G: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    x0: np.ndarray
    P0: np.ndarray
    Psi: np.ndarray = None

    @cached_property
    def R_inv(self) -> np.ndarray:
        return np.linalg.inv(self.R)

    @property
    def state_dim(self) -> int:
        return self.x0.shape[0]


class MatrixGradient:
    """
    Градиент критерия ММП по матрицам модели

    Property
    ---------------
    F, Q, R, x0, P0
        Градиент по соответсвующим матрицам
    y
        Список градиентов по измерениям на каждом шаге
    """
    MATRIX_NAMES = ("F", "Q", "R", "x0", "P0")

    def __init__(self, matrices: ModelMatrices, steps: int):
        n = matrices.state_dim
        noise_dim = at_step(matrices.G, 0).shape[1]
        self.F = np.zeros_like(at_step(matrices.F, 0), dtype=float)
        self.Q = np.zeros((noise_dim, noise_dim))
        self.R = np.zeros_like(matrices.R, dtype=float)
        self.x0 = np.zeros((n, 1))
        self.P0 = np.zeros((n, n))
        self.y = steps * [None]

    def to_parameters(self, derivatives: list[dict]) -> np.ndarray:
        """
        Переход к скалярным параметрам
        dL/dtheta

        :param derivatives: список по параметрам (каждый элемент - словарт)

        :return: вектор dL/dtheta
        """
        result = np.zeros(len(derivatives))
        for i, derivs in enumerate(derivatives):
            for name, dM in derivs.items():
                if name not in self.MATRIX_NAMES:
                    raise KeyError(f'Неизвестная матрица модели: {name}\n')
                result[i] += float(np.tensordot(getattr(self, name), dM, axes=2))
        return result

DERIVATIVE_NAMES = ("F", "H", "G", "Q", "R", "x0", "P0")

def complete_derivatives(matrices: ModelMatrices, derivatives: list[dict]) -> list[dict]:
    """
    Дополнение производных модели нулями: для каждого неизвестного параметра
    возвращается словарь со всеми матрицми модели. Если матрица не зависит от
    параметра, то возвращаем ноль. 

    :param matrices: матрицы модели
    :param derivatives: список по параметрам из model.derivatives
    :return: список по параметрам
    """
    zeros = {
        "F": np.zeros(at_step(matrices.F, 0).shape),
        "H": np.zeros(at_step(matrices.H, 0).shape),
        "G": np.zeros(at_step(matrices.G, 0).shape),
        "Q": np.zeros(matrices.Q.shape),
        "R": np.zeros(matrices.R.shape),
        "x0": np.zeros(matrices.x0.shape),
        "P0": np.zeros(matrices.P0.shape),
    }

    result = []
    for derivs in derivatives:
        for name in derivs:
            if name not in DERIVATIVE_NAMES:
                raise KeyError(f'Неизвестная матрица модели: {name}\n')
        result.append({name: derivs.get(name, zeros[name]) for name in DERIVATIVE_NAMES})
    return result

class ModelBase(ABC):
    """
    Шаблонный класс параметризованной модели.

    Property
    ---------------
    parametr_names 
        Имена параметров тетта 
    """

    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        ...
    
    @abstractmethod
    def matrices(self, theta: np.ndarray) -> ModelMatrices:
        """
        Матрицы модели при заданных параметрах.

        :param theta: вектор неизвестных параметров
        :return: матрицы модели
        """

    @abstractmethod
    def derivatives(self, theta: np.ndarray) -> list[dict]:
        """
        Проивзодные матриц модели по каждому неизвестному
        параметру.

        :param theta: вектор неизвестных параметров
        :return: список по параметроам, элемент - словарь 
        {"F": dF/dTheta_i, ...}
        """
    