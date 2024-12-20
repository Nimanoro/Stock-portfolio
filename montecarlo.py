import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def simulate_random_returns(mean, std, periods, iterations):
    return np.random.lognormal(mean, std, size=(iterations, periods))

def simulate_portfolio_growth(initial_investment, simulated_returns):
    return initial_investment * np.cumprod(simulated_returns, axis=1)

def monte_carlo_simulation(historical_returns, initial_value, years, iterations):
    """Perform Monte Carlo simulation."""
    aggregated_returns = historical_returns.mean(axis=1)

    mean = np.log1p(aggregated_returns.mean()) 
    std = aggregated_returns.std()
    days = years * 252

    simulated_returns = simulate_random_returns(mean, std, days, iterations)
    return simulate_portfolio_growth(initial_value, simulated_returns)


def plot_monte_carlo(simulated_growth):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(simulated_growth.T, color='orange', alpha=0.05)
    percentiles = np.percentile(simulated_growth, [10, 50, 90], axis=0)

    ax.plot(percentiles[1], color='blue', label='50th Percentile (Median)')
    ax.fill_between(range(simulated_growth.shape[1]), percentiles[0], percentiles[2], color='blue', alpha=0.2, label='10th-90th Percentile')
    ax.set_title("Monte Carlo Simulation", fontsize=16)
    ax.set_xlabel("Days", fontsize=14)
    ax.set_ylabel("Portfolio Value", fontsize=14)
    ax.legend()
    plt.show()

def summarize_simulation(simulated_growth):
    final_values = simulated_growth[:, -1]
    summary = {
        "Median Portfolio Value": np.median(final_values),
        "10th Percentile": np.percentile(final_values, 10),
        "90th Percentile": np.percentile(final_values, 90)
    }
    return summary
