import numpy as np
import pandas as pd
from scipy.optimize import minimize
from calibrate import loss_function, calculate_ask_price
import matplotlib.pyplot as plt
from utils import display_order_book, to_float, to_json, extract_trades
from tqdm import tqdm
from orderBook import OrderBook, DollarsAndShares, PriceSizePairs, DynamicOrderBook
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from numba import njit, prange
import statsmodels.api as sm
plt.rcParams['figure.figsize'] = (14, 8)

def load_order_books(df, start_idx, end_idx):
    bids = []
    asks = []
    for i in range(start_idx, end_idx):
        bids.append([DollarsAndShares(dollars=x[0], shares=x[1]) for x in to_float(to_json(df.bids.iloc[i]))])
        asks.append([DollarsAndShares(dollars=x[0], shares=x[1]) for x in to_float(to_json(df.asks.iloc[i]))])
    return [DynamicOrderBook(descending_bids=bids[i], ascending_asks=asks[i]) for i in range(len(bids))]

def calibrate_lambda(order_books):
    trades = extract_trades(order_books)

    x_list = []
    y_list = []

    for t, x in trades:
        book = order_books[t]
        V_t = book.mid_price()
        A_t = book.ask_price()
        s_t = book.bid_ask_spread()

        # Перманентный импакт = A_t - V_t - s/2
        y = A_t - V_t - s_t / 2
        x_list.append(x)
        y_list.append(y)

    x_array = np.array(x_list).reshape(-1, 1)
    y_array = np.array(y_list)

    if len(x_array) == 0:
        return 1.0  # fallback

    # robust линейная регрессия
    X_design = sm.add_constant(x_array)
    model = sm.RLM(y_array, X_design, M=sm.robust.norms.HuberT())
    results = model.fit()
    return results.params[1]

def calibrate_r(trade_times, trade_volumes, asks, s, V_t, k):
    trade_times = np.array(trade_times, dtype=np.float64)
    trade_volumes = np.array(trade_volumes, dtype=np.float64)
    asks = np.array(asks, dtype=np.float64)

    def loss_r_only(r_param):
        modeled_ask_prices = np.array([
            calculate_ask_price(t, trade_times, trade_volumes, k, r_param[0], s, V_t)[0]
            for t in trade_times
        ])
        return np.mean((asks - modeled_ask_prices) ** 2)

    result = minimize(
        loss_r_only,
        [0.1],
        bounds=[(1e-4, 10000.0)],
        method='L-BFGS-B'
    )

    r_opt = result.x[0]
    return r_opt

def calibrate_k_params(train_book, real_book, r, dt=1.0):
    """
    Калибрует параметры восстановления объёмов заявки (k_exp, k_lin) по ошибке
    между восстановленным и реальным стаканом.
    
    Параметры:
        train_book: последний стакан из трейна (OrderBook)
        real_book: реальный стакан после восстановления (OrderBook)
        r: параметр восстановления
        dt: временной шаг между train и real

    Возвращает:
        (k_exp, k_lin): оптимальные параметры восстановления
    """
    def loss_k(params):
        k_exp, k_lin = params
        restored = train_book.restore_order_book(r=r, k_exp=k_exp, k_lin=k_lin, dt=dt)

        depth = min(len(real_book.ascending_asks), len(restored.ascending_asks), 5)
        real_asks = np.array([ds.shares for ds in real_book.ascending_asks[:depth]])
        model_asks = np.array([ds.shares for ds in restored.ascending_asks[:depth]])

        real_bids = np.array([ds.shares for ds in real_book.descending_bids[:depth]])
        model_bids = np.array([ds.shares for ds in restored.descending_bids[:depth]])

        ask_error = np.mean((real_asks - model_asks) ** 2)
        bid_error = np.mean((real_bids - model_bids) ** 2)

        return ask_error + bid_error

    result = minimize(
        loss_k,
        x0=[np.random.uniform(1.0, 10.0), np.random.uniform(0.1, 1.0)],
        bounds=[(0.01, 50.0), (0.0, 10.0)],
        method='L-BFGS-B',
        options={'maxiter': 100}
    )

    return result.x  # k_exp, k_lin




# def calibrate_lambda(order_books):
#     trades = extract_trades(order_books)
#     trade_times = [t[0] for t in trades]
#     trade_volumes = [t[1] for t in trades]

#     mid_prices = np.array([order_books[i].mid_price() for i in trade_times])
#     X_cum = np.cumsum(trade_volumes)
#     V0 = mid_prices[0]
#     X0 = X_cum[0]
#     y = mid_prices - V0
#     X = X_cum - X0

#     X_design = sm.add_constant(X)
#     model = sm.RLM(y, X_design, M=sm.robust.norms.HuberT())
#     results = model.fit()
#     # print(results.summary())
#     return results.params[1]

# def calibrate_r(trade_times, trade_volumes, asks, s, V_t, k):
    
#     trade_times = np.array(trade_times, dtype=np.float64)
#     trade_volumes = np.array(trade_volumes, dtype=np.float64)
    
#     def loss_r_only(r_param):

#         modeled_ask_prices = np.array([
#             calculate_ask_price(t, trade_times, trade_volumes, k, r_param[0], s, V_t)[0]
#             for t in trade_times
#         ])
#         true_min = np.min(asks)
#         true_max = np.max(asks)
#         asks_scaled = (asks - true_min) / (true_max - true_min)
#         modeled_scaled = (modeled_ask_prices - true_min) / (true_max - true_min)
#         return np.mean((asks_scaled - modeled_scaled) ** 2)
    
#     result = minimize(
#         loss_r_only,
#         [0.1],
#         bounds=[(1e-9, 70)],
#         method='trust-constr'
#     )
#     r_opt = result.x[0]
#     loss_center = loss_r_only([r_opt])
#     loss_left = loss_r_only([r_opt - 0.01])
#     loss_right = loss_r_only([r_opt + 0.01])

#     # print(f"r_opt = {r_opt:.4f}")
#     # print(f"Loss @ r_opt = {loss_center:.5e}")
#     # print(f"Loss @ r_opt ± 0.01 = {loss_left:.5e}, {loss_right:.5e}")
#     return result.x[0]

def simulate_mid_price_paths(order_books, n_sim=1000, mu=0.0, sigma=0.5, T=1.0):
    """
    Генерация траекторий mid-price F_t с броуновским движением.

    Parameters:
        order_books : список order book’ов (ожидается, что у них есть .mid_price())
        n_sim       : количество симуляций
        mu          : дрейф броуновского движения
        sigma       : волатильность
        T           : горизонт моделирования (в тех же единицах, что и частота стакана)

    Returns:
        F_result : массив (n_sim, N+1) — симулированные траектории
        F_mean   : средняя траектория F_t (усреднение по симам)
        t_grid   : массив времени (N+1,)
    """
    F0 = order_books[0].mid_price()
    N = len(order_books) - 1
    dt = T / N
    t_grid = np.linspace(0, T, N+1)

    F_result = np.empty((n_sim, N+1), dtype=np.float64)

    for sim in tqdm(range(n_sim), desc="Simulating F_t paths"):
        z = np.random.randn(N)
        F = np.empty(N+1, dtype=np.float64)
        F[0] = F0
        for i in range(1, N+1):
            F[i] = F[i-1] + mu * dt + sigma * np.sqrt(dt) * z[i-1]
        F_result[sim, :] = F

    F_mean = np.mean(F_result, axis=0)
    return F_result, F_mean, t_grid

@njit(parallel=True)
def simulate_X_t(n_sim, t_grid, trades_history):
    """
    Для каждого симуляционного запуска и для каждого момента времени из t_grid
    вычисляет кумулятивную сумму сделок (сумма x_val для сделок с tau <= t).
    
    Parameters:
      n_sim : int
          Число симуляций.
      t_grid : 1D numpy array (np.float64)
          Моменты времени, для которых считается кумулятивная сумма.
      trades_history : 2D numpy array (np.float64) с формой (n_trades, 2)
          История сделок, где trades_history[:, 0] – tau, а trades_history[:, 1] – x_val.
          
    Returns:
      X_res : 2D numpy array (n_sim, len(t_grid))
          Для каждой симуляции массив кумулятивных сумм.
    """
    n_time = t_grid.shape[0]
    n_trades = trades_history.shape[0]
    X_res = np.empty((n_sim, n_time), dtype=np.float64)
    
    # Для каждой симуляции
    for sim in prange(n_sim):
        # Массив для хранения кумулятивных сумм для текущей симуляции
        X_local = np.empty(n_time, dtype=np.float64)
        # Обычно в начальный момент (t_grid[0]) кумулятивная сумма равна 0
        X_local[0] = 0.0
        # Для каждого момента времени, начиная с 1 (так как для 0 уже задано)
        for i in range(1, n_time):
            cum_sum = 0.0
            current_time = t_grid[i]
            # Проходим по истории сделок
            for j in range(n_trades):
                tau = trades_history[j, 0]
                if tau <= current_time:
                    cum_sum += trades_history[j, 1]
                else:
                    break  # trades_history отсортирован по времени tau
            X_local[i] = cum_sum
        X_res[sim, :] = X_local
    return X_res

@njit
def compute_A_single(t_grid, V, s, lmbd, r, trades_history):
    N = t_grid.shape[0]
    A = np.empty(N, dtype=np.float64)
    for i in range(N):
        t_current = t_grid[i]
        sum_term = 0.0
        for j in range(trades_history.shape[0]):
            tau = trades_history[j, 0]
            if tau <= t_current:
                time_diff = t_current - tau
                sum_term += trades_history[j, 1] * np.exp(-r * time_diff)
            else:
                break
        A[i] = V[i] + s/2.0 + (1.0 / lmbd) * sum_term
    return A

@njit(parallel=True)
def simulate_A(n_sim, t_grid, V, s, lmbd, r, trades_history):
    N = t_grid.shape[0]
    A_result = np.empty((n_sim, N), dtype=np.float64)
    for sim in prange(n_sim):
        A_result[sim, :] = compute_A_single(t_grid, V, s, lmbd, r, trades_history)
    
    return A_result

def simulate_full_surface(order_books, lmbd, r, s, n_sim=1000, mu=0.0, sigma=0.5, T=1.0):
    F_result, F, t_grid = simulate_mid_price_paths(order_books, n_sim=n_sim, mu=mu, sigma=sigma, T=T)

    # Трейды
    trades = extract_trades(order_books)
    trades_history = np.array([(t_grid[t_idx], x_val) for (t_idx, x_val) in trades], dtype=np.float64)

    # X_t
    X_res = simulate_X_t(n_sim, t_grid, trades_history)
    X_t = np.mean(X_res, axis=0)

    # V_t
    V = F + lmbd * X_t

    # A_t
    A_result = simulate_A(n_sim, t_grid, V, s, lmbd, r, trades_history)
    A = np.mean(A_result, axis=0)

    return A, t_grid


def run_model(df, train_idx, val_idx):
    # 1. OrderBooks
    train_books = load_order_books(df, *train_idx)
    val_books = load_order_books(df, *val_idx)

    # 2. Lambda
    lmbd = calibrate_lambda(train_books)

    # 3. r
    trades = extract_trades(train_books)
    trade_times = [t[0] for t in trades]
    trade_volumes = [t[1] for t in trades]
    asks = np.array([train_books[i].ask_price() for i in trade_times])
    s = train_books[0].bid_ask_spread()
    V_t = train_books[0].mid_price()
    r = calibrate_r(trade_times, trade_volumes, asks, s, V_t, 1/abs(lmbd))

    # 4. Моделируем F, X, A (как у тебя)
    A = simulate_full_surface(val_books, lmbd, r, s)

    return A, val_books
