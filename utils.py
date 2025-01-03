import matplotlib.pyplot as plt
import numpy as np
import json


def display_order_book(
    ask: np.ndarray, 
    bid: np.ndarray,
) -> None:
    """
    Рисует ордербук, размерности а и б такие: a, b = (prices * volumes)

    Args:
        ask (np.ndarray): Asks array. Asks prices = ask[:, 0], Asks volumes = ask[:, 1]
        bid (np.ndarray): Bids array. Bids prices = bid[:, 0], Bids volumes = bid[:, 1]
    """
 

    bid_prices = bid[:, 0]
    bid_shares = bid[:, 1]

    plt.bar(bid_prices, bid_shares, color='blue', width=0.01)

    ask_prices = ask[:, 0]
    ask_shares = ask[:, 1]
    
    plt.bar(ask_prices, ask_shares, color='red', width=0.01)

    # all_prices = sorted(bid_prices + ask_prices)
    # all_ticks = ["%d" % x for x in all_prices]
    # plt.xticks(all_prices, all_ticks)
    plt.grid(axis='y')
    plt.xlabel("Prices")
    plt.ylabel("Number of Shares")
    plt.title("Order Book")
    # plt.xticks(x_pos, x)
    plt.show()
    

def to_json(
    str: str
) -> list:
    """
    Convert the JSON string to a Python object.

    Args:
        str (str): _description_

    Returns:
        list: _description_
    """
    cleaned_string = str.replace('\\', '').strip('"')

    # 
    data = json.loads(cleaned_string)
    
    return data

def to_float(
    array: list
) -> list:
    
    result_arr = np.zeros((len(array), 2))
    
    for i in range(len(result_arr)):
        
        result_arr[i][0] = float(array[i]['price']['units']) + float(array[i]['price']['nano']) / 1e9
        result_arr[i][1] = float(array[i]['quantity'])
    
    return result_arr
