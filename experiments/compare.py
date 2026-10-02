"""
Сравнение методов вычисления градиента: точность и время
python3 -m experiments.compare
"""

import time
import numpy as np
import torch

from experiments.generate_data import simulate_measurements
from filters.KF import KalmanFilter
from gradients.backward import BackwardGradient
from gradients.forward import ForwardSensivityGradient
from gradients.finite_diff import FiniteDifferenceGradient
from gradients.automatic import AutomaticDiff
from models.postition import Position


def best_time(call, repeats: int = 5) -> float:
    """
    Минимальное время (берется миниамльное из 5 запусков)

    :param call: функция без аргументов
    :param repeats: число запусков
    :return: время в секундах
    """
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        timings.append(time.perf_counter() - start)
    return min(timings)

def main():
    rng = np.random.default_rng(0)
    N = 500

    model = Position(dt=1)
    theta_true = np.array([0.98, 0.02, 1.0])
    y = simulate_measurements(model, theta_true, N, np.array([[0.0], [0.0], [0.0], [0.0]]), rng)

    theta = np.array([0.5, 0.10, 0.10])
    kf = KalmanFilter()

    # ref = FiniteDifferenceGradient(kf, eps = 1e-6)
    ref = AutomaticDiff(KalmanFilter(xp = torch))

    methods = [ref, BackwardGradient(kf), ForwardSensivityGradient(kf), FiniteDifferenceGradient(kf, eps = 1e-6)]

    results = {m.name: m.value_and_gradient(model, theta, y) for m in methods}
    L, g_ref = results[ref.name]
    print(f"L = {L:.6f}\n")
    header = f"{'параметр':<10}" + "".join(f"{m.name:>16}" for m in methods)
    header += "".join(f"{'ошибка (' + m.name + ')':>24}" for m in methods if m is not ref)
    print(header)
    for i, name in enumerate(model.parameter_names):
        row = f"{name:<10}" + "".join(f"{results[m.name][1][i]:>16.8f}" for m in methods)
        for m in methods:
            if m is ref:
                continue
            rel = abs(results[m.name][1][i] - g_ref[i]) / max(abs(g_ref[i]), 1e-12)
            row += f"{rel:>24.2e}"
        print(row)

    # --- время ---
    timings = {m.name: best_time(lambda m=m: m.value_and_gradient(model, theta, y))
               for m in methods}
    fastest = min(timings.values())

    print(f"\nвремя ({len(theta)} параметра, N = {N} шагов):")
    for name, t in timings.items():
        print(f"  {name:<12}{t * 1e3:>10.1f} мс   (x{t / fastest:.1f})")


if __name__ == "__main__":
    main()
