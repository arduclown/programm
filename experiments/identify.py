"""
Идентификация параметров модели по измерениям 
Запуск из папки, где лежит kalman_gradients:
    python3 -m experiments.identify
"""
import numpy as np
from scipy.optimize import minimize

from experiments.generate_data import simulate_measurements
from filters.KF import KalmanFilter
from gradients.backward import BackwardGradient
from gradients.forward import ForwardSensivityGradient
from gradients.finite_diff import FiniteDifferenceGradient

from models.postition import Position


def main():
    rng = np.random.default_rng(0)

    model = Position(dt=1)
    theta_true = np.array([0.98, 0.02, 1.0])
    y = simulate_measurements(model, theta_true, 500, np.array([[0.0], [0.0], [0.0], [0.0]]), rng)

    method = ForwardSensivityGradient(KalmanFilter())

    result = minimize(lambda theta: method.value_and_gradient(model, theta, y),
                      x0=np.array([0.5, 0.10, 0.10]), jac=True, method="L-BFGS-B",
                      bounds=[(0.0, 2.0), (1e-6, None), (1e-6, None)])

    # print(result)
    print("истина :", theta_true)
    print("оценка :", result.x)
    print("ошибка :", result.x - theta_true)


if __name__ == "__main__":
    main()
