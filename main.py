import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from montecarlo import simulate_random_returns, simulate_portfolio_growth, monte_carlo_simulation, plot_monte_carlo, summarize_simulation, monte_carlo_simulation_scenario
from scipy.optimize import minimize
# Convert stock tickers input to a list of uppercase tickers
def convert_input_to_tickers(input_str):
    tickers = [ticker.upper() for ticker in input_str.split()]
    return tickers

# Convert weights input to a list of floats and normalize if necessary
def convert_weights_to_list(input_str):
    weights = list(map(float, input_str.split()))
    total = sum(weights)
    if total > 1:
        weights = [weight / total for weight in weights]
    return weights

# Fetch historical stock data
def get_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        if data.empty:
            st.error("No data retrieved. Please check your tickers or date range.")
            return None
        return data
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Plot adjusted close prices
def plot_data(data):
    fig, ax = plt.subplots(figsize=(10, 7))
    for column in data['Adj Close']:
        ax.plot(data['Adj Close'][column], label=column)
    ax.set_title('Adjusted Close Price', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price', fontsize=14)
    ax.legend()
    st.pyplot(fig)

# Calculate portfolio value over time
def calculate_portfolio_value(data, weights, initial_investment):
    adj_close = data['Adj Close']
    normalized_prices = adj_close / adj_close.iloc[0]  # Normalize to the first day's price
    portfolio_value = normalized_prices.dot(weights) * initial_investment
    return portfolio_value

def optimize_portfolio(data, initial_investment):
    returns = data['Adj Close'].pct_change().dropna()
    mean_returns = returns.mean()
    corr_matrix = returns.corr()
    if (corr_matrix > 0.99).sum().sum() > len(corr_matrix):
        st.error("Highly correlated stocks may cause issues in optimization.")
        return

    cov_matrix = returns.cov()

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(mean_returns)))
    initial_guess = [1. / len(mean_returns)] * len(mean_returns)

    result = minimize(portfolio_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Calculate portfolio metrics
def calculate_portfolio_metrics(portfolio_value, benchmark_value, start_date, end_date):
    start_value = portfolio_value.iloc[0]
    end_value = portfolio_value.iloc[-1]
    total_return = (end_value - start_value) / start_value
    days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    annualized_return = (1 + total_return) ** (365 / days) - 1
    daily_returns = portfolio_value.pct_change().dropna()
    benchmark_returns = benchmark_value.pct_change().dropna()
    portfolio_risk = daily_returns.std()
    sharpe_ratio = (total_return - 0.03) / portfolio_risk if portfolio_risk > 0 else 0
    max_drawdown = ((portfolio_value.cummax() - portfolio_value) / portfolio_value.cummax()).max()
    tracking_error = ((daily_returns - benchmark_returns).std()) if not benchmark_returns.empty else None
    beta = (daily_returns.corr(benchmark_returns) * daily_returns.std() / benchmark_returns.std()) if not benchmark_returns.empty else None
    alpha = (annualized_return - (0.03 + beta * (benchmark_returns.mean() * 252 - 0.03))) if beta is not None else None
    sortino_ratio = (total_return - 0.03) / daily_returns[daily_returns < 0].std() if (daily_returns[daily_returns < 0].std() > 0) else None

    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else None



    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Portfolio Risk": portfolio_risk,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Tracking Error": tracking_error,
        "Beta": beta,
        "Alpha": alpha,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio
    }

# Calculate Value at Risk (VaR)
def calculate_var(portfolio_value, confidence_level=0.95):
    daily_returns = portfolio_value.pct_change().dropna()
    var = -daily_returns.quantile(1 - confidence_level)
    return var


# Display simulation summary
def summarize_simulation(simulated_growth):
    final_values = simulated_growth[:, -1]
    return {
        "Median Portfolio Value": np.median(final_values),
        "10th Percentile": np.percentile(final_values, 10),
        "90th Percentile": np.percentile(final_values, 90)
    }

def plot_correlation_heatmap(data):
    """Plot correlation heatmap."""
    try:
        adj_close = data['Adj Close']
        if adj_close.shape[1] < 2:
            st.error("Correlation heatmap requires at least two stocks.")
            return
        
        # Calculate correlation matrix
        corr = adj_close.pct_change().dropna().corr()

        # Ensure no empty correlation matrix
        if corr.empty:
            st.error("Correlation matrix could not be computed.")
            return

        # Create a mask for the upper triangle
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": 0.75})
        ax.set_title("Correlation Heatmap", fontsize=16)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while plotting the correlation heatmap: {e}")

# Streamlit Application
def run_streamlit_app():
    st.title("Stock Portfolio Simulator")

    page = st.sidebar.radio("Select Page", ["Portfolio Backtesting", "Monte Carlo Simulation"])

    if page == "Portfolio Backtesting":
        st.header("Portfolio Backtesting")

        # Sidebar inputs
        tickers_input = st.sidebar.text_input("Stock Tickers (space-separated):")
        weights_input = st.sidebar.text_input("Weights (space-separated):")
        initial_investment = st.sidebar.number_input("Initial Investment:", min_value=0.0, value=1000.0)
        start_date = st.sidebar.date_input("Start Date")
        end_date = st.sidebar.date_input("End Date")

        if st.sidebar.button("Fetch and Analyze"):
            tickers = convert_input_to_tickers(tickers_input)
            weights = convert_weights_to_list(weights_input)

            if len(tickers) != len(weights):
                st.error("The number of tickers and weights must match.")
                return

            data = get_data(tickers, start_date, end_date)
            benchmark_data = get_data(["^GSPC"], start_date, end_date)  # S&P 500 as benchmark

            if data is not None and benchmark_data is not None:
                st.write("## Portfolio Performance")
                plot_data(data)

                portfolio_value = calculate_portfolio_value(data, weights, initial_investment)
                benchmark_value = calculate_portfolio_value(benchmark_data, [1.0], initial_investment)
                optimized_weights = optimize_portfolio(data, initial_investment)

                # Store portfolio value for later use
                st.session_state['portfolio_value'] = portfolio_value

                # Plot portfolio performance
                fig, ax = plt.subplots(figsize=(10, 7))
                ax.plot(portfolio_value, label='Portfolio Value', color='orange')
                ax.plot(benchmark_value, label='Benchmark (S&P 500)', color='blue', linestyle='--')
                ax.set_title('Portfolio vs Benchmark Performance', fontsize=16)
                ax.set_xlabel('Date', fontsize=14)
                ax.set_ylabel('Value', fontsize=14)
                ax.legend()
                st.pyplot(fig)

                # Calculate and display metrics
                metrics = calculate_portfolio_metrics(portfolio_value, benchmark_value, start_date, end_date)
                metrics["Value at Risk (VaR)"] = calculate_var(portfolio_value)
                st.write("## Portfolio Metrics")
                st.table(pd.DataFrame(metrics, index=["Value"]).T)

                st.write("## Correlation Matrix")
                plot_correlation_heatmap(data)
                st.write("## Optimized Weights")
                st.write(pd.DataFrame(optimized_weights, index=tickers, columns=["Weight"]))


    elif page == "Monte Carlo Simulation":
        st.header("Monte Carlo Simulation with Scenario Analysis")

        # Sidebar inputs
        tickers_input = st.sidebar.text_input("Stock Tickers (space-separated):")
        years_of_history = st.sidebar.number_input("Years of Historical Data:", min_value=1, value=5)
        initial_investment = st.sidebar.number_input("Initial Portfolio Value:", min_value=0.0, value=10000.0)
        simulation_years = st.sidebar.number_input("Years to Simulate:", min_value=1, value=5)
        iterations = st.sidebar.number_input("Number of Iterations:", min_value=100, value=1000)

        # Scenario selection
        st.sidebar.header("Scenario Analysis")

        scenarios = {
            "Bull Market": {"mean": 0.0005, "std": 0.01},  # Higher returns, lower volatility
            "Bear Market": {"mean": -0.0003, "std": 0.02},  # Negative returns, higher volatility
            "Stagnant Market": {"mean": 0.0001, "std": 0.005},  # Neutral returns, low volatility
        }

        scenario = st.sidebar.selectbox("Select Scenario", list(scenarios.keys()) + ["Custom"])

        if scenario == "Custom":
            custom_mean = st.sidebar.number_input("Custom Mean (daily return)", value=0.0001, step=0.0001)
            custom_std = st.sidebar.number_input("Custom Std Dev (daily volatility)", value=0.01, step=0.001)
        else:
            selected_scenario = scenarios[scenario]
            custom_mean, custom_std = selected_scenario["mean"], selected_scenario["std"]

        if st.sidebar.button("Run Simulation"):
            tickers = convert_input_to_tickers(tickers_input)

            if not tickers:
                st.error("Please input valid stock tickers.")
                return

            # Fetch historical data for validation
            end_date = pd.Timestamp.today()
            start_date = end_date - pd.DateOffset(years=years_of_history)
            data = get_data(tickers, start_date, end_date)

            if data is not None:
                # Run the scenario-based Monte Carlo simulation
                simulated_growth = monte_carlo_simulation_scenario(
                    initial_value=initial_investment,
                    years=simulation_years,
                    iterations=iterations,
                    mean=custom_mean,
                    std=custom_std
                )

                # Display simulation results
                st.write(f"## Monte Carlo Simulation Results ({scenario})")
                plot_monte_carlo(simulated_growth)
                summary = summarize_simulation(simulated_growth)
                st.write("### Simulation Summary")
                st.json(summary)


if __name__ == "__main__":
    run_streamlit_app()
