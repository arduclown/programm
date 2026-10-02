import numpy as np

from core.model import ModelBase, ModelMatrices

class Position(ModelBase):
    def __init__(self, dt: float = 1.0, x0: np.ndarray = None, P0: np.ndarray = None):
        self._dt = dt
        self._x0 = np.zeros((4, 1)) if x0 is None else x0
        self._P0 = np.diag([100.0, 100.0, 10.0, 10.0]) if P0 is None else P0

    @property
    def parameter_names(self) -> list[str]:
        return ["theta1", "theta2", "theta3"]
    
    def matrices(self, theta: np.ndarray) -> ModelMatrices:
        theta1, theta2, theta3 = theta
        dt = self._dt
        # Матрица перехода состояний F (4x4)
        F = np.array([
            [1.0, 0.0, dt,   0.0],
            [0.0, 1.0, 0.0,  dt ],
            [0.0, 0.0, theta1, 0.0],
            [0.0, 0.0, 0.0, theta1]
        ])
        
        # Матрица измерений H (2x4)
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        # Матрица G (4x2), умножается на шум w(t_k)
        G = np.array([
            [dt**2 / 2.0, 0.0],
            [0.0, dt**2 / 2.0],
            [dt, 0.0],
            [0.0, dt]
        ])
        
        # Ковариация шума процесса Q (2x2)
        Q = theta2 * np.eye(2)
        
        # Ковариация шума измерений R (2x2)
        R = theta3 * np.eye(2)

        return ModelMatrices(
            F=F,
            H=H,
            G=G,
            Q=Q,
            R=R,
            x0=self._x0,
            P0=self._P0,
        )
    
    def derivatives(self, theta: np.ndarray) -> list[dict]:
        # Производная по theta1 (входит в F)
        dF_dtheta1 = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Производная по theta2 (входит в Q)
        dQ_dtheta2 = np.eye(2)

        # Производная по theta3 (входит в R)
        dR_dtheta3 = np.eye(2)

        return [
            {"F": dF_dtheta1},   # по theta1
            {"Q": dQ_dtheta2},   # по theta2
            {"R": dR_dtheta3},   # по theta3
        ]