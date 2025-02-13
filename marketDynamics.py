import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from utils import display_order_book, to_float, to_json, extract_trades
from tqdm import tqdm
import statsmodels.api as sm
from scipy.optimize import minimize
from typing import Optional, Tuple, List
from orderBook import OrderBook, DollarsAndShares, PriceSizePairs, DynamicOrderBook
from sklearn.preprocessing import MinMaxScaler, StandardScaler
plt.rcParams['figure.figsize'] = (14, 8)


@dataclass
class MarketDynamics:
    
    order_books: list[DynamicOrderBook]
    f0: float # init value
    sigma: float # volatility
    t: float # modeling time
    n: int # count time steps
    trades_volumes: list[float] # trades volumes $x_{t_i}$
    
    lambda_estimate_: Optional[float] = None  # оценка λ
    spread: Optional[float] = None # oderbook spread
    k: Optional[float] = None           # оптимизированный параметр k
    r: Optional[float] = None           # оптимизированный параметр r
    wiener_process: Optional[np.ndarray] = None
    ft: Optional[np.ndarray] = None
    v_t: Optional[np.ndarray] = None # mid-quote prices on each time step
    trades_til_n: Optional[np.ndarray] = None
    times_til_n: Optional[np.ndarray] = None
    asks: Optional[np.ndarray] = None   # истинные ask цены (из order_books)
    time_points: Optional[np.ndarray] = None
    
    def simulate_brownian_motion(
        self,
    ) -> Tuple[float, float]:
        """
        Симулирует чистое броуновское движение для F_t (без дрейфа).
        
        Args:
            F0 (float): Начальное значение F_0.
            sigma (float): Волатильность (\sigma).
            T (float): Общее время моделирования.
            N (int): Количество временных шагов.

        Returns:
            np.ndarray: Значения F_t для каждого временного момента.
        """
        dt = self.t / self.n  # Шаг времени
        time_points = np.linspace(0, self.t, self.n)
        
        # Генерация стандартного броуновского движения
        self.wiener_process = np.random.normal(0, np.sqrt(dt), size=self.n).cumsum()
        
        # Вычисление F_t (без дрейфа) source obizhaeva wang
        self.ft = self.f0 + self.sigma * self.wiener_process
        return time_points, self.ft
    
    def lambda_estimate(
        self
    ) -> float:
        """
        Оценка коэффициента постоянного ценового импакта (\lambda) через линейную регрессию.
        
        Args:
            V (np.ndarray): Массив mid-quote цен (V_t) на каждом шаге времени.
            trades (list[float]): Объёмы сделок (x_{t_i}).
            F0 (float): Начальная фундаментальная стоимость (F_t, предполагается постоянной).
        
        Returns:
            float: Оценка параметра \lambda.
        """
        # difference from fundament price
        mid_prices = np.array([self.order_books[i].mid_price() for i in self.times_til_n])
        self.v_t = self.order_books[0]
        y = mid_prices - self.f0
        
        # cumulative volume of trades
        x = np.cumsum(self.trades_volumes)
        
        X = sm.add_constant(x)
        
        model = sm.OLS(y, X).fit_regularized(
            method='elastic_net'
        )
        
        self.lambda_estimate_ = model.params[1]
        
        if self.info:
            pass # TODO ADD MODEL DESCRIPTION
              
        return self.lambda_estimate_
        
        
    def simulate_mid_price(
        self,
    ) -> np.ndarray:
        
        trades = extract_trades(self.order_books)
        
        trades_til_n = []
        times_til_n = []
        
        for i in range(len(trades)):
            
            if trades[i][0] < self.n:
                trades_til_n.append(trades[i][1])
                times_til_n.append(trades[i][0])
                
        self.trades_til_n = np.array(trades_til_n)
        self.times_til_n = np.array(times_til_n)
        self.spread = self.order_books[0].bid_ask_spread()
        self.lambda_estimate_ = self.lambda_estimate()
        # Симуляция
        self.time_points, self.f_t = self.simulate_brownian_motion()
        self.v_t = self.ft + self.lambda_estimate_ * np.cumsum(trades_til_n)
        
        return self.v_t
    
    def calculate_ask_price(
        self
    ):
        """
        
        """
        self.spread = self.order_books[0].bid_ask_spread()
        impact_sum = np.sum(
            self.trades_til_n * self.k * np.exp(-self.r * (self.t - self.times_til_n) * (self.t >= self.times_til_n))
        )
        
        return self.v_t + (self.spread / 2) + impact_sum, impact_sum
    
    def _loss_function(
        self,
        init_params: list,
        v_t,
        t,
        times_til_n,
        trades_til_n,
        spread
    ):
        k, r = init_params
        v_t = self.simulate_mid_price()
        modeled_ask_prices = np.array([
            self.calculate_ask_price(
            )[0] for _ in self.times_til_n
        ])
        
        scaler = MinMaxScaler()
        true_ask_prices_sc = scaler.fit_transform(self.asks.reshape(-1, 1)) # ???
        modeled_ask_prices_sc = scaler.fit_transform(modeled_ask_prices.reshape(-1, 1))
        return np.mean((true_ask_prices_sc - modeled_ask_prices_sc)**2)
    
    
    def minimize(
        self,
    ):
        
        mid_price = self.order_books[0].mid_price()
        init_params = [0.01, 0.1]
        result = minimize(
            self._loss_function,
            init_params,
            args=(self.times_til_n, self.trades_til_n, np.array(self.asks), self.spread, mid_price),
            bounds=[(1e-9, 20), (1e-9, 70)],  # Constraints for k and r
            method='L-BFGS-B'
        )
        self.k, self.r = result.x
        
        return self.k, self.r
        