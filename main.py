import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

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

# Streamlit Application
def run_streamlit_app():
    st.title("Stock Portfolio Simulator")

    tickers_input = st.text_input("Stock Tickers (space-separated):")
    start_date = st.text_input("Start Date (YYYY-MM-DD):")
    end_date = st.text_input("End Date (YYYY-MM-DD):")
    initial_investment = st.number_input("Initial Investment:", min_value=0.0, value=1000.0)
    weights_input = st.text_input("Weights (space-separated):")

    if st.button("Fetch and Plot"):
        tickers = convert_input_to_tickers(tickers_input)
        weights = convert_weights_to_list(weights_input)

        data = get_data(tickers, start_date, end_date)
        if data is not None:
            st.write("### Portfolio Performance")
            plot_data(data)

            portfolio_value = calculate_portfolio_value(data, weights, initial_investment)

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.plot(portfolio_value, label='Portfolio Value', color='orange')
            ax.set_title('Portfolio Performance', fontsize=16)
            ax.set_xlabel('Date', fontsize=14)
            ax.set_ylabel('Value', fontsize=14)
            ax.legend()

            st.pyplot(fig)

if __name__ == "__main__":
    run_streamlit_app()
