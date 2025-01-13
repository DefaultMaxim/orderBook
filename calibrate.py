import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize


def calculate_ask_price(
    t: int,
    trade_times: np.ndarray,
    trade_volumes: np.ndarray,
    k: float,
    r: float,
    s: float,
    V_t
):
    """
    Рассчитываем теоретический аск-прайс в момент времени t,
    используя фиксированный k и подбираемый r.
    """
    impact_sum = np.sum(trade_volumes * k * np.exp(-r * (t - trade_times) * (t >= trade_times)))
    return V_t + (s / 2) + impact_sum, impact_sum

def loss_function(
    r_param: float,
    trade_times: np.ndarray,
    trade_volumes: np.ndarray,
    true_ask_prices: np.ndarray,
    s: float,
    V_t,
    k
):
    """
    Функция потерь для оценки параметра r при фиксированном k.
    """
    r = r_param[0]
    
    modeled_ask_prices = np.array([
        calculate_ask_price(t, trade_times, trade_volumes, k, r, s, V_t)[0] for t in trade_times
    ])
    
    scaler = MinMaxScaler()
    true_ask_prices_sc = scaler.fit_transform(true_ask_prices.reshape(-1, 1))
    modeled_ask_prices_sc = scaler.fit_transform(modeled_ask_prices.reshape(-1, 1))
    return np.mean((true_ask_prices_sc - modeled_ask_prices_sc)**2)