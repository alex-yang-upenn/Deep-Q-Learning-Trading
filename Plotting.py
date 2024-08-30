import numpy as np
import math

import matplotlib.pyplot as plt


def formatPrice(n):
    """
    Formats the price of a stock

    Args:
        Price

    Returns:
        Price with correct sign, formatted to two decimal places
    """
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


def PlotBehavior(dataset, states_buy, states_sell, profit):
    """
    Visualizes the daily prices(red) alongside the Agent's buy(purple) and sell(black) decisions
    as a function of time in days.

    Args:
        dataset: Sequential daily stock prices
        states_buy: Indices where the Agent decided to buy.
        states_sell: Indices where the Agent decided to sell.
        profit: The total profit achieved by the agent.

    Returns:
        None: This function displays a plot.
    """
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(dataset, color='r', lw=2.)
    ax.plot(dataset, '^', markersize=10, color='m', label="Buying signal", markevery=states_buy)
    ax.plot(dataset, 'v', markersize=10, color='k', label="Selling signal", markevery=states_sell)

    ax.set_title(f'Total gains: {profit}')
    ax.legend()

    plt.show()