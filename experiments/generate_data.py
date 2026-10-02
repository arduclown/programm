import numpy as np

from core.model import ModelBase
from core.matrix_utils import at_step

def simulate_measurements(model: ModelBase, theta: np.ndarray, steps: int,
                        x_start: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """
    Моделирование системы и измерений
    :param model: модель
    :param theta: истинные параметры модели
    :param steps: число шагов N
    :param x_start: истинное начальное состояние
    :param rng: генератор случайных чисел

    :return: список измерений y
    """
    matrices = model.matrices(theta)
    Q_root = np.linalg.cholesky(matrices.Q)
    R_root = np.linalg.cholesky(matrices.R)
    noise_dim, meas_dim = Q_root.shape[0], R_root.shape[0]

    x = x_start.copy()
    y = []
    for k in range(steps):
        F, G, H = at_step(matrices.F, k), at_step(matrices.G, k), at_step(matrices.H, k)
        x = F @ x + G @ (Q_root @ rng.standard_normal((noise_dim, 1)))
        y.append(H @ x + R_root @ rng.standard_normal((meas_dim, 1)))
    return y