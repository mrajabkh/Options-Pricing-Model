from scipy.stats import norm
import math
import numpy as np
import matplotlib.pyplot as plt


def calculateBS(S, K, T, r, sigma):
    """
    S - Current stock price
    K - Strike price
    T - Time to expire in yrs
    r - risk free interest
    sigma - volatility of the stock (std of returns)
    """

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return call_price, put_price

def pnl_plot(S,K, call_price, put_price):

    # Range of possible stock prices at expiry
    S = np.linspace(S*0.5, S*1.5, int(S*2))

    # --- Payoffs at expiry ---
    call_payoff = np.maximum(0, S - K)
    put_payoff = np.maximum(0, K - S)

    # --- Profits (payoff - premium) ---
    call_profit = call_payoff - call_price
    put_profit = put_payoff - put_price

    # Plotting
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(S, call_profit, label="Call Profit", color="blue")
    ax.plot(S, put_profit, label="Put Profit", color="red")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Option Payoff & Profit Diagrams K={K}")
    ax.set_xlabel("Stock Price at Expiry ($S$)")
    ax.set_ylabel("Profit / Payoff")
    ax.legend()
    ax.grid(True)
    plt.show()
    return fig, ax

def call_pnl_heatmap(K, S, call_price):
    """
    K : strike price
    put_price : fixed put price (optional, can add later)
    S : array of stock prices
    call_price : array of call premiums
    """
    # Create grid
    S = np.linspace(S*0.5, S*1.5, int(S*2))
    call_price = np.linspace(call_price*0.5, call_price*1.5, int(call_price*2))

    S_grid, C_grid = np.meshgrid(S, call_price)
    
    # Calculate P&L
    call_pnl = np.maximum(0, S_grid - K) - C_grid
    
    # Plot heatmap
    plt.figure(figsize=(10,6))
    heatmap = plt.pcolormesh(S_grid, C_grid, call_pnl, shading='auto', cmap='RdYlGn')
    plt.colorbar(heatmap, label="Call Profit")
    plt.xlabel("Stock Price at Expiry")
    plt.ylabel("Call Premium Paid")
    plt.title(f"Call Option P&L Heatmap (K={K})")
    plt.show()

def put_pnl_heatmap(K, S, put_price):
    """
    K : strike price
    S : current stock price (used to scale the plot)
    put_price : put premium paid
    """
    # Dynamic ranges
    S_range = np.linspace(S*0.5, S*1.5, int(S*2))
    put_price_range = np.linspace(put_price*0.5, put_price*1.5, int(put_price*2))

    # Create grid
    S_grid, P_grid = np.meshgrid(S_range, put_price_range)
    
    # Calculate P&L
    put_pnl = np.maximum(0, K - S_grid) - P_grid
    
    # Plot heatmap
    plt.figure(figsize=(10,6))
    heatmap = plt.pcolormesh(S_grid, P_grid, put_pnl, shading='auto', cmap='RdYlGn')
    plt.colorbar(heatmap, label="Put Profit")
    plt.xlabel("Stock Price at Expiry")
    plt.ylabel("Put Premium Paid")
    plt.title(f"Put Option P&L Heatmap (K={K})")
    plt.show()

def option_vol_heatmap(K, S0, T, r, option_type="call"):
    """
    Heatmap of option P&L vs volatility
    """
    # Ranges
    vol_range = np.linspace(0.01, 1.0, 100)      # 1% to 100% volatility
    S_range   = np.linspace(S0*0.5, S0*1.5, 100) # expiry prices
    
    # Mesh grid
    S_grid, vol_grid = np.meshgrid(S_range, vol_range)

    # Premiums for each volatility
    if option_type == "call":
        premiums = np.array([calculateBS(S0, K, T, r, sigma)[0] for sigma in vol_range])
    else:
        premiums = np.array([calculateBS(S0, K, T, r, sigma)[1] for sigma in vol_range])
    
    premium_grid = np.tile(premiums.reshape(-1, 1), (1, len(S_range)))

    # Payoff at expiry
    if option_type == "call":
        payoff = np.maximum(S_grid - K, 0)
    else:
        payoff = np.maximum(K - S_grid, 0)

    # Profit/Loss
    pnl = payoff - premium_grid

    # Plot heatmap
    plt.figure(figsize=(10,6))
    heatmap = plt.pcolormesh(S_grid, vol_grid, pnl, shading="auto", cmap="RdYlGn")
    plt.colorbar(heatmap, label="Profit / Loss")
    plt.xlabel("Stock Price at Expiry")
    plt.ylabel("Volatility (σ)")
    plt.title(f"{option_type.capitalize()} Option P&L vs Volatility (K={K}, T={T}y)")
    plt.show()

def monte_carlo_call_pnl(S0, K, T, r, sigma, premium=None, n_sims=100000):
    if premium is None:
        premium, _ = calculateBS(S0, K, T, r, sigma)
    
    # Simulate expiry stock prices
    Z = np.random.randn(n_sims)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Compute payoff and PnL
    payoff = np.maximum(ST - K, 0)
    pnl = payoff - premium
    return pnl, ST, premium

def plot_stock_distribution(ST, S0):
    # Calculate percentiles for main histogram
    lower, upper = np.percentile(ST, [1, 99])
    
    plt.figure(figsize=(10,5))
    plt.hist(ST, bins=50, range=(lower, upper), density=True, alpha=0.7, color="skyblue", label="99% of simulations")
    
    # Mark extremes
    extreme_low = ST[ST < lower]
    if len(extreme_low) > 0:
        plt.scatter(extreme_low, [0]*len(extreme_low), color="red", marker="v", label="Extreme low")
    extreme_high = ST[ST > upper]
    if len(extreme_high) > 0:
        plt.scatter(extreme_high, [0]*len(extreme_high), color="green", marker="^", label="Extreme high")
    
    plt.axvline(S0, color="r", linestyle="--", label="Initial Price")
    plt.title("Simulated Stock Prices at Expiry (Monte Carlo)")
    plt.xlabel("Stock Price at Expiry")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def plot_pnl_distribution(pnl, S0, premium):
    lower, upper = np.percentile(pnl, [1, 99])
    plt.figure(figsize=(10,5))
    plt.hist(pnl, bins=50, range=(lower, upper), density=True, alpha=0.7, color="skyblue", label="99% of simulations")
    
    # Extreme values
    extreme_low = pnl[pnl < lower]
    extreme_high = pnl[pnl > upper]
    if len(extreme_low) > 0:
        plt.scatter(extreme_low, [0]*len(extreme_low), color="red", marker="v", label="Extreme loss")
    if len(extreme_high) > 0:
        plt.scatter(extreme_high, [0]*len(extreme_high), color="green", marker="^", label="Extreme gain")
        
    plt.title("Monte Carlo Call Option PnL Distribution")
    plt.xlabel("Profit / Loss")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def run_call_simulation(S0, K, T, r, sigma, n_sims=100000):
    call_price, put_price = calculateBS(S0, K, T, r, sigma)
    pnl, ST, premium = monte_carlo_call_pnl(S0, K, T, r, sigma, n_sims=n_sims)
    
    print(f"Black Scholes Call Price: {call_price:.4f}")
    print(f"Monte Carlo estimated PnL = {np.mean(pnl):.4f}")
    
    plot_stock_distribution(ST,S0)
    plot_pnl_distribution(pnl, S0, premium)


# Example usage
# pnl_plot(100,100,10.45,5.57)
# call_pnl_heatmap(100, 100, 10.45)
# put_pnl_heatmap(100, 100, 5.57)
# option_vol_heatmap(K=100, S0=100, T=1, r=0.02, option_type="call")
# option_vol_heatmap(K=100, S0=100, T=1, r=0.02, option_type="put")
# run_call_simulation(S0=100, K=100, T=1, r=0.05, sigma=0.2, n_sims=100000)