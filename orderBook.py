from __future__ import annotations

import numpy as np
from dataclasses import dataclass, replace
from typing import Sequence, Tuple, Optional, List
from pprint import pprint
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class DollarsAndShares:

    dollars: float
    shares: int


PriceSizePairs = Sequence[DollarsAndShares]


@dataclass(frozen=True)
class OrderBook:
    
    descending_bids: PriceSizePairs
    ascending_asks: PriceSizePairs

    def bid_price(self) -> float:
        """
        Возвращает текущую цену покупки (bid price), 
        то есть самую высокую цену среди заявок на покупку.
        """
        
        return self.descending_bids[0].dollars

    def ask_price(self) -> float:
        """
        Возвращает текущую цену продажи (ask price), 
        то есть самую низкую цену среди заявок на продажу.
        """
        
        return self.ascending_asks[0].dollars

    def mid_price(self) -> float:
        """
        Возвращает среднюю цену (mid price), рассчитанную как 
        среднее арифметическое между ценой покупки и ценой продажи.
        """
        
        return (self.bid_price() + self.ask_price()) / 2

    def bid_ask_spread(self) -> float:
        """
        Возвращает разницу между ценой продажи (ask price) 
        и ценой покупки (bid price).
        """
        
        return self.ask_price() - self.bid_price()

    def market_depth(self) -> float:
        """
        Возвращает глубину рынка, которая определяется как 
        разница между самой низкой ценой продажи (ask) и 
        самой низкой ценой покупки (bid).
        """
        
        return self.ascending_asks[-1].dollars - \
            self.descending_bids[-1].dollars
            
    @staticmethod
    def eat_book(
        ps_pairs: PriceSizePairs,
        shares: int
    ) -> Tuple[DollarsAndShares, PriceSizePairs]:
        '''
        Returned DollarsAndShares represents the pair of
        dollars transacted and the number of shares transacted
        on ps_pairs (with number of shares transacted being less
        than or equal to the input shares).
        Returned PriceSizePairs represents the remainder of the
        ps_pairs after the transacted number of shares have eaten into
        the input ps_pairs.
        '''
        rem_shares: int = shares
        dollars: float = 0.
        for i, d_s in enumerate(ps_pairs):
            this_price: float = d_s.dollars
            this_shares: int = d_s.shares
            dollars += this_price * min(rem_shares, this_shares)
            if rem_shares < this_shares:
                return (
                    DollarsAndShares(dollars=dollars, shares=shares),
                    [DollarsAndShares(
                        dollars=this_price,
                        shares=this_shares - rem_shares
                    )] + list(ps_pairs[i+1:])
                )
            else:
                rem_shares -= this_shares

        return (
            DollarsAndShares(dollars=dollars, shares=shares - rem_shares),
            []
        )
        
    def sell_limit_order(self, price: float, shares: int) -> \
            Tuple[DollarsAndShares, OrderBook]:
        """
        Размещает лимитный ордер на продажу.
        
        Args:
            price (float): Цена продажи.
            shares (int): Количество акций для продажи.

        Returns:
            Tuple[DollarsAndShares, OrderBook]: 
                - DollarsAndShares: количество акций и денег, обработанных в сделке.
                - OrderBook: обновлённый ордербук после размещения ордера.
        """
        index: Optional[int] = next((i for i, d_s
                                     in enumerate(self.descending_bids)
                                     if d_s.dollars < price), None)
        eligible_bids: PriceSizePairs = self.descending_bids \
            if index is None else self.descending_bids[:index]
        ineligible_bids: PriceSizePairs = [] if index is None else \
            self.descending_bids[index:]

        d_s, rem_bids = OrderBook.eat_book(eligible_bids, shares)
        new_bids: PriceSizePairs = list(rem_bids) + list(ineligible_bids)
        rem_shares: int = shares - d_s.shares

        if rem_shares > 0:
            new_asks: List[DollarsAndShares] = list(self.ascending_asks)
            index1: Optional[int] = next((i for i, d_s
                                          in enumerate(new_asks)
                                          if d_s.dollars >= price), None)
            if index1 is None:
                new_asks.append(DollarsAndShares(
                    dollars=price,
                    shares=rem_shares
                ))
            elif new_asks[index1].dollars != price:
                new_asks.insert(index1, DollarsAndShares(
                    dollars=price,
                    shares=rem_shares
                ))
            else:
                new_asks[index1] = DollarsAndShares(
                    dollars=price,
                    shares=new_asks[index1].shares + rem_shares
                )
            return d_s, OrderBook(
                ascending_asks=new_asks,
                descending_bids=new_bids
            )
        else:
            return d_s, replace(
                self,
                descending_bids=new_bids
            )

    def sell_market_order(
        self,
        shares: int
    ) -> Tuple[DollarsAndShares, OrderBook]:
        """
        Размещает рыночный ордер на продажу.

        Args:
            shares (int): Количество акций для продажи.

        Returns:
            Tuple[DollarsAndShares, OrderBook]: 
                - DollarsAndShares: количество акций и денег, обработанных в сделке.
                - OrderBook: обновлённый ордербук после размещения ордера.
        """
        
        d_s, rem_bids = OrderBook.eat_book(
            self.descending_bids,
            shares
        )
        return (d_s, replace(self, descending_bids=rem_bids))

    def buy_limit_order(self, price: float, shares: int) -> \
            Tuple[DollarsAndShares, OrderBook]:
        """
        Размещает лимитный ордер на покупку.

        Args:
            price (float): Желаемая цена покупки.
            shares (int): Количество акций для покупки.

        Returns:
            Tuple[DollarsAndShares, OrderBook]: 
                - DollarsAndShares: количество акций и денег, обработанных в сделке.
                - OrderBook: обновлённый ордербук после размещения ордера.
        """        
        
        index: Optional[int] = next((i for i, d_s
                                     in enumerate(self.ascending_asks)
                                     if d_s.dollars > price), None)
        eligible_asks: PriceSizePairs = self.ascending_asks \
            if index is None else self.ascending_asks[:index]
        ineligible_asks: PriceSizePairs = [] if index is None else \
            self.ascending_asks[index:]

        d_s, rem_asks = OrderBook.eat_book(eligible_asks, shares)
        new_asks: PriceSizePairs = list(rem_asks) + list(ineligible_asks)
        rem_shares: int = shares - d_s.shares

        if rem_shares > 0:
            new_bids: List[DollarsAndShares] = list(self.descending_bids)
            index1: Optional[int] = next((i for i, d_s
                                          in enumerate(new_bids)
                                          if d_s.dollars <= price), None)
            if index1 is None:
                new_bids.append(DollarsAndShares(
                    dollars=price,
                    shares=rem_shares
                ))
            elif new_bids[index1].dollars != price:
                new_bids.insert(index1, DollarsAndShares(
                    dollars=price,
                    shares=rem_shares
                ))
            else:
                new_bids[index1] = DollarsAndShares(
                    dollars=price,
                    shares=new_bids[index1].shares + rem_shares
                )
            return d_s, replace(
                self,
                ascending_asks=new_asks,
                descending_bids=new_bids
            )
        else:
            return d_s, replace(
                self,
                ascending_asks=new_asks
            )

    def buy_market_order(
        self,
        shares: int
    ) -> Tuple[DollarsAndShares, OrderBook]:
        """
        Размещает рыночный ордер на покупку.

        Args:
            shares (int): Количество акций для покупки.

        Returns:
            Tuple[DollarsAndShares, OrderBook]: 
                - DollarsAndShares: количество акций и денег, обработанных в сделке.
                - OrderBook: обновлённый ордербук после размещения ордера.
        """
        
        d_s, rem_asks = OrderBook.eat_book(
            self.ascending_asks,
            shares
        )
        return (d_s, replace(self, ascending_asks=rem_asks))
    
    def pretty_print_order_book(
        self
    ) -> None:
        """
        Печатает текущее состояние ордербука в читабельном формате.
        """
        
        print()
        print("Bids")
        pprint(self.descending_bids)
        print()
        print("Asks")
        print()
        pprint(self.ascending_asks)
        print()

    def display_order_book(
        self,
    ) -> None:
        """
        Рисует ордербук, размерности а и б такие: a, b = (prices * volumes)
        
        График показывает:
        - Цены заявок по оси X.
        - Количество акций по оси Y.

        Args:
            ask (np.ndarray): Asks array. Asks prices = ask[:, 0], Asks volumes = ask[:, 1]
            bid (np.ndarray): Bids array. Bids prices = bid[:, 0], Bids volumes = bid[:, 1]
        """
    

        bid_prices = [d_s.dollars for d_s in self.descending_bids]
        bid_shares = [d_s.shares for d_s in self.descending_bids]
        
        if self.descending_bids:
            plt.bar(bid_prices, bid_shares, color='blue', width=0.008, label='Покупки')

        ask_prices = [d_s.dollars for d_s in self.ascending_asks]
        ask_shares = [d_s.shares for d_s in self.ascending_asks]
        
        if self.ascending_asks:
            plt.bar(ask_prices, ask_shares, color='red', width=0.008, label='Продажи')

        # all_prices = sorted(bid_prices + ask_prices)
        # all_ticks = ["%d" % x for x in all_prices]
        # plt.xticks(all_prices, all_ticks)
        plt.grid(axis='y')
        plt.xlabel("Цены", fontsize=15)
        plt.legend(fontsize=15)
        plt.ylabel("Объемы", fontsize=15)
        # plt.title("Order Book")
        # plt.xticks(x_pos, x)
        plt.show()
        
# RUNNING EXAMPLE
# ob0: OrderBook = OrderBook(descending_bids=bids, ascending_asks=asks)
# ob0.pretty_print_order_book()
# ob0.display_order_book()

# print("Sell Limit Order of (107, 40)")
# print()
# d_s1, ob1 = ob0.sell_limit_order(119, 40)
# proceeds1: float = d_s1.dollars
# shares_sold1: int = d_s1.shares
# print(f"Sales Proceeds = {proceeds1:.2f}, Shares Sold = {shares_sold1:d}")
# # ob1.pretty_print_order_book()
# ob1.display_order_book()

# print("Sell Market Order of 120")
# print()
# d_s2, ob2 = ob1.sell_market_order(120)
# proceeds2: float = d_s2.dollars
# shares_sold2: int = d_s2.shares
# print(f"Sales Proceeds = {proceeds2:.2f}, Shares Sold = {shares_sold2:d}")
# # ob2.pretty_print_order_book()
# ob2.display_order_book()

# print("Buy Limit Order of (100, 80)")
# print()
# d_s3, ob3 = ob2.buy_limit_order(121, 80)
# bill3: float = d_s3.dollars
# shares_bought3: int = d_s3.shares
# print(f"Purchase Bill = {bill3:.2f}, Shares Bought = {shares_bought3:d}")
# # ob3.pretty_print_order_book()
# ob3.display_order_book()

# print("Sell Limit Order of (104, 60)")
# print()
# d_s4, ob4 = ob3.sell_limit_order(123, 60)
# proceeds4: float = d_s4.dollars
# shares_sold4: int = d_s4.shares
# print(f"Sales Proceeds = {proceeds4:.2f}, Shares Sold = {shares_sold4:d}")
# # ob4.pretty_print_order_book()
# ob4.display_order_book()

# print("Buy Market Order of 150")
# print()
# d_s5, ob5 = ob4.buy_market_order(130)
# bill5: float = d_s5.dollars
# shares_bought5: int = d_s5.shares
# print(f"Purchase Bill = {bill5:.2f}, Shares Bought = {shares_bought5:d}")
# # ob5.pretty_print_order_book()
# ob5.display_order_book()

@dataclass(frozen=True)
class DynamicOrderBook(OrderBook):

    def restore_order_book(
        self,
        r: float,
        k_exp: float,
        k_lin: float,
        dt: float
    ) -> OrderBook:
        """
        Восстанавливает книгу заявок по динамике, описанной в статье:
        экспоненциальное восстановление ликвидности.

        Args:
            r (float): Скорость восстановления книги (resilience).
            k_exp (float): Экспоненциальный коэффициент восстановления.
            k_lin (float): Линейный коэффициент восстановления.
            dt (float): Время, прошедшее с последнего восстановления.

        Returns:
            OrderBook: Обновлённая книга заявок после восстановления.
        """
        # Восстановление бидов
        restored_bids = [
            DollarsAndShares(
                dollars=d_s.dollars,
                shares=int(d_s.shares + k_exp * np.exp(-r * dt) + k_lin * dt)
            )
            for d_s in self.descending_bids
        ]

        # Восстановление асков
        restored_asks = [
            DollarsAndShares(
                dollars=d_s.dollars,
                shares=int(d_s.shares + k_exp * np.exp(-r * dt) + k_lin * dt)
            )
            for d_s in self.ascending_asks
        ]

        return OrderBook(
            descending_bids=restored_bids,
            ascending_asks=restored_asks
        )

    def execute_dynamic_order(
        self,
        order_type: str,
        price: Optional[float],
        shares: int,
        r: float,
        k_exp: float,
        k_lin: float,
        dt: float
    ) -> Tuple[DollarsAndShares, OrderBook, OrderBook]:
        """
        Исполняет ордер и восстанавливает книгу заявок после исполнения.

        Args:
            order_type (str): Тип ордера ('buy_limit', 'buy_market', 'sell_limit', 'sell_market').
            price (float): Цена для лимитного ордера (опционально для рыночных).
            shares (int): Количество акций для исполнения.
            r (float): Скорость восстановления книги.
            k_exp (float): Экспоненциальный коэффициент восстановления объёма.
            k_lin (float): Линейный коэффициент восстановления объёма.
            dt (float): Интервал времени восстановления.

        Returns:
            Tuple[DollarsAndShares, OrderBook, OrderBook]: 
                - Исполненный ордер,
                - Книга заявок после исполнения (до восстановления),
                - Книга заявок после восстановления.
        """
        if order_type == 'buy_limit':
            result, updated_book = self.buy_limit_order(price, shares)
        elif order_type == 'buy_market':
            result, updated_book = self.buy_market_order(shares)
        elif order_type == 'sell_limit':
            result, updated_book = self.sell_limit_order(price, shares)
        elif order_type == 'sell_market':
            result, updated_book = self.sell_market_order(shares)
        else:
            raise ValueError("Invalid order type. Use 'buy_limit', 'buy_market', 'sell_limit', or 'sell_market'.")

        restored_book = updated_book.restore_order_book(r=r, k_exp=k_exp, k_lin=k_lin, dt=dt)
        return result, updated_book, restored_book
    
    def extract_trades(
        historical_order_books: List[OrderBook]
    ) -> List[Tuple[int, float]]:
        """
        Извлекает сделки из исторических данных стаканов.

        Args:
            historical_order_books (List[OrderBook]): Список исторических ордербуков.

        Returns:
            List[Tuple[int, float]]: Список сделок [(время, объем сделки)].
        """
        trades = []

        # Проходим по всем моментам времени
        for t in range(1, len(historical_order_books)):
            prev_book = historical_order_books[t - 1]
            current_book = historical_order_books[t]

            # Анализируем изменения в объемах асков
            for prev_ask, current_ask in zip(prev_book.ascending_asks, current_book.ascending_asks):
                if prev_ask.dollars == current_ask.dollars:
                    diff = prev_ask.shares - current_ask.shares
                    if diff > 0:  # Продажа
                        trades.append((t, float(diff)))

            # Анализируем изменения в объемах бидов
            for prev_bid, current_bid in zip(prev_book.descending_bids, current_book.descending_bids):
                if prev_bid.dollars == current_bid.dollars:
                    diff = current_bid.shares - prev_bid.shares
                    if diff > 0:  # Покупка
                        trades.append((t, float(diff)))

        return trades
