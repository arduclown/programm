from dataclasses import dataclass

import numpy as np

from filters.base import FilterBase
from core.model import ModelMatrices, MatrixGradient
from core.matrix_utils import at_step
from core.likelihood import Likelihood

@dataclass
class PredictRecord:
    """ Величины прогноза """
    x_prev: np.ndarray #x_{k|k}
    P_prev: np.ndarray #P_{k|k}

@dataclass
class CorrectRecord:
    K: np.ndarray # коэффициент усиления
    eps: np.ndarray # невязка
    B_inv: np.ndarray # обратная матрица к ковариации невязки
    I_minus_KH: np.ndarray # I - KH

class KalmanFilter(FilterBase):
    """
    Класс реализации ковариационногр фильтра Калмана
    """
    def __init__(self, xp = np):
        self._xp = xp
        self._likelihood = Likelihood(xp)
    
    # ---- ПРЯМОЙ ШАГ (ФИЛЬТРАЦИЯ ----
    def predict(self, state, matrices: ModelMatrices, k: int, u_k: np.ndarray = None):
        x_post, P_post = state
        F, G = at_step(matrices.F, k), at_step(matrices.G, k)

        # x_{k+1|k} = F_{k+1} @ x_{k|k} + Psi_{k+1} @ u_{k+1}
        x_pred = F @ x_post
        if matrices.Psi is not None and u_k is not None:
            x_pred = x_pred + at_step(matrices.Psi, k) @ u_k
        #P_{k+1|k} = F_{k+1} @ P_{k|k} @ F_{k+1} ^ T + G_{k+1} @ Q_{k+1} @ G_{k+1}^T
        P_pred = F @ P_post @ F.T + G @ matrices.Q @ G.T

        return (x_pred, P_pred), PredictRecord(x_prev=x_post, P_prev=P_post)
    
    def correct(self, predicted_state, y_k: np.ndarray, matrices: ModelMatrices, k: int):
        x_pred, P_pred = predicted_state
        H = at_step(matrices.H, k)

        eps = y_k - H @ x_pred
        B = H @ P_pred @ H.T + matrices.R
        B_inv = self._xp.linalg.inv(B)
        K = P_pred @ H.T @ B_inv
        I_minus_KH = self._xp.eye(matrices.state_dim) - K @ H

        x_post = x_pred + K @ eps
        P_post = I_minus_KH @ P_pred

        loss_k = self._likelihood.value(eps, B, B_inv)
        record = CorrectRecord(K=K, eps=eps, B_inv=B_inv, I_minus_KH=I_minus_KH)
        return (x_post, P_post), loss_k, record
    
    # ---- ОБРАТНЫЙ ШАГ (ВЫЧИСЛЕНИЕ ГРАДИЕНТА) ----
    def correct_backward(self, grad_after, record: CorrectRecord, matrices: ModelMatrices,
                        k: int, grads: MatrixGradient):
        gx, gP = grad_after
        H = at_step(matrices.H, k)
        K, eps, B_inv, IKH = record.K, record.eps, record.B_inv, record.I_minus_KH

        # производные слагаемого l_{k+1|k} формулы (28-30)
        dl_deps, dl_dB = self._likelihood.gradient(eps, B_inv)

        # параметры текущего шага
        A = K.T @ gx @ eps.T @ B_inv
        grads.R += K.T @ gP @ K - 0.5 * (A + A.T) + dl_dB # 26
        grads.y[k] = K.T @ gx + dl_deps # 24

        # переход к прогнозным величинам
        C = gx @ eps.T @ matrices.R_inv @ H
        gP_pred = IKH.T @ (gP + 0.5 * (C + C.T)) @ IKH + H.T @ dl_dB @ H # 21
        gx_pred = IKH.T @ gx - H.T @ dl_deps #21 (!!!!!!тут минус, потому что dl_deps записана
                                             #    с плюсом!!!!!)
        return gx_pred, gP_pred

    
    def predict_backward(self, grad_after, record: PredictRecord, matrices: ModelMatrices,
                        k: int, grads: MatrixGradient):
        gx_pred, gP_pred = grad_after
        F, G = at_step(matrices.F, k), at_step(matrices.G, k)

        grads.Q += G.T @ gP_pred @ G # 25

        # производная по матрице перехода (вклад через x и P)
        grads.F += gx_pred @ record.x_prev.T + 2.0 * gP_pred @ F @ record.P_prev

        # шаг назад
        gx = F.T @ gx_pred # 22
        gP = F.T @ gP_pred @ F # 23
        return gx, gP


    # ---- СТАНДАРТНОЕ ВЫЧСИЛЕНИЕ ГРАДИЕНТА ----
    def predict_forward(self, sens, state, matrices: ModelMatrices,
                        derivs: dict, k: int):
        dx, dP = sens
        x, P = state
        F, G, Q = at_step(matrices.F, k), at_step(matrices.G, k), at_step(matrices.Q, k)
        dF, dG, dQ = derivs["F"], derivs["G"], derivs["Q"]

        dP_pred = dF @ P @ F.T + F @ dP @ F.T + (dF @ P @ F.T).T + G @ dQ @ G.T + dG @ Q @ G.T + (dG @ Q @ G.T).T

        dx_pred = dF @ x + F @ dx

        return dx_pred, dP_pred
    
    def correct_forward(self, sens_pred, predicted_state, record: CorrectRecord,
                        matrices: ModelMatrices, derivs: dict, k: int):
        dx_pred, dP_pred = sens_pred
        x_pred, P_pred = predicted_state
        H = at_step(matrices.H, k)
        dH, dR = derivs["H"], derivs["R"]
        K, eps, B_inv, I_minus_KH = record.K, record.eps, record.B_inv, record.I_minus_KH

        deps = -dH @ x_pred - H @ dx_pred
        dH_P_HT = dH @ P_pred @ H.T
        dB = H @ dP_pred @ H.T + dH_P_HT + dH_P_HT.T + dR     

        dK = (dP_pred @ H.T + P_pred @ dH.T - K @ dB) @ B_inv

        dx_post = dx_pred + dK @ eps + K @ deps
        dP_post = I_minus_KH @ dP_pred - (dK @ H + K @ dH) @ P_pred

        dloss_k = self._likelihood.derivative(eps, B_inv, deps, dB)
        return (dx_post, dP_post), dloss_k