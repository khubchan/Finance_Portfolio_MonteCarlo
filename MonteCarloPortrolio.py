
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---- Inputs for Anil----
initial_portfolio = 2_400_000 #Change this number for starting Portfolio
annual_expenses = 67_500 # assume 3% withdrwal rule to use funds for 30+ years.
inflation = 0.03        # 2.5% inflation
years = 30 #how many years to withdraw the funds.
n_sims = 500000 # higher the number, longer it will take to run MonteCarlo calculations.
risk_free_rate = 0.02    # 2% assumed
np.random.seed(59)

# Example portfolio 1 allocation
allocations = {
    "Equities": 0.48,
    "Bonds": 0.075,
    "REITs": 0.01,
    "CEFs": 0.39,
    "Cash": 0.037
}

#RESULTS: Portfolio Expected Return: 6.91%
#RESULTS: Portfolio Volatility: 9.04%
#RESULTS: Sharpe Ratio: 0.543
#RESULTS: Sortino Ratio: 0.958
#RESULTS: Max Drawdown: -15.93%
#Also generates retirement_simulation.xlsx Excel file.

# Example portfolio #2 allocation
#allocations = {
#    "Equities": 0.75,
#    "Bonds": 0.18,
#    "REITs": 0.00,
#    "CEFs": 0.01,
#    "Cash": 0.04
#}
#RESULTS: Portfolio Expected Return: 6.13%
#RESULTS: Portfolio Volatility: 11.29%
#RESULTS: Sharpe Ratio: 0.366
#RESULTS: Sortino Ratio: 0.689
#RESULTS: Max Drawdown: -28.58%


# Expected returns (annualized)
expected_returns = {
    "Equities": 0.07,
    "Bonds": 0.04,
    "REITs": 0.06,
    "CEFs": 0.08,
    "Cash": 0.02
}

# Volatilities (annualized std dev)
volatilities = {
    "Equities": 0.15,
    "Bonds": 0.05,
    "REITs": 0.12,
    "CEFs": 0.14,
    "Cash": 0.01
}

assets = list(allocations.keys())
returns_vector = np.array([expected_returns[a] for a in assets])
vol_vector = np.array([volatilities[a] for a in assets])
weights_current = np.array([allocations[a] for a in assets])

# ---- Current Portfolio Metrics ----
port_return = np.dot(weights_current, returns_vector)
port_vol = np.sqrt(np.dot(weights_current**2, vol_vector**2))  # simplified
sharpe_ratio = (port_return - risk_free_rate) / port_vol

# Simulate monthly returns for Sortino & Drawdown
months = years * 12
sim_returns = np.random.normal(port_return/12, port_vol/np.sqrt(12), months)
cum_returns = (1 + sim_returns).cumprod()

downside = sim_returns[sim_returns < risk_free_rate/12]
downside_dev = downside.std() * np.sqrt(12)
sortino_ratio = (port_return - risk_free_rate) / downside_dev if downside_dev > 0 else np.nan

rolling_max = np.maximum.accumulate(cum_returns)
drawdown = (cum_returns - rolling_max) / rolling_max
max_drawdown = drawdown.min()

print(f"Portfolio Expected Return: {port_return:.2%}")
print(f"Portfolio Volatility: {port_vol:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
print(f"Sortino Ratio: {sortino_ratio:.3f}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# ---- Monte Carlo Simulation (same as before) ----
final_balances = []
all_paths = []

for sim in range(n_sims):
    balance = initial_portfolio
    expense = annual_expenses
    yearly_balances = []
    for year in range(years):
        annual_r = np.random.normal(port_return, port_vol)
        balance = balance * (1 + annual_r) - expense
        expense *= (1 + inflation)
        yearly_balances.append(max(balance, 0))
        if balance <= 0:
            yearly_balances.extend([0] * (years - len(yearly_balances)))
            break
    all_paths.append(yearly_balances)
    final_balances.append(balance)

final_balances = np.array(final_balances)
all_paths = np.array(all_paths)

# ---- Percentile Table ----
percentiles = [5, 10, 25, 50, 75, 85, 90]
percentile_table = np.percentile(all_paths, percentiles, axis=0)

df_percentiles = pd.DataFrame({
    "Year": np.arange(1, years+1),
    "5th %ile": percentile_table[0],
    "10th %ile": percentile_table[1],
    "25th %ile": percentile_table[2],
    "Median": percentile_table[3],
    "75th %ile": percentile_table[4],
    "85th %ile": percentile_table[5],
    "90th %ile": percentile_table[6]
})

# ---- Efficient Frontier with Sharpe & Sortino ----
n_portfolios = 10000
weights_list, returns_list, vols_list, sharpe_list, sortino_list = [], [], [], [], []

for _ in range(n_portfolios):
    w = np.random.random(len(assets))
    w /= np.sum(w)
    port_r = np.dot(w, returns_vector)
    port_v = np.sqrt(np.dot(w**2, vol_vector**2))  # simplified
    
    # Simulate monthly returns for downside deviation
    sim_r = np.random.normal(port_r/12, port_v/np.sqrt(12), months)
    downside_r = sim_r[sim_r < risk_free_rate/12]
    downside_dev_r = downside_r.std() * np.sqrt(12) if len(downside_r) > 0 else np.nan
    
    sharpe = (port_r - risk_free_rate) / port_v
    sortino = (port_r - risk_free_rate) / downside_dev_r if downside_dev_r > 0 else np.nan
    
    weights_list.append(w)
    returns_list.append(port_r)
    vols_list.append(port_v)
    sharpe_list.append(sharpe)
    sortino_list.append(sortino)

returns_list = np.array(returns_list)
vols_list = np.array(vols_list)
sharpe_list = np.array(sharpe_list)
sortino_list = np.array(sortino_list)
weights_list = np.array(weights_list)

# Max Sharpe Portfolio
idx_max_sharpe = np.argmax(sharpe_list)
w_max_sharpe = weights_list[idx_max_sharpe]
ret_max_sharpe = returns_list[idx_max_sharpe]
vol_max_sharpe = vols_list[idx_max_sharpe]
sharpe_max = sharpe_list[idx_max_sharpe]

# Max Sortino Portfolio
idx_max_sortino = np.nanargmax(sortino_list)
w_max_sortino = weights_list[idx_max_sortino]
ret_max_sortino = returns_list[idx_max_sortino]
vol_max_sortino = vols_list[idx_max_sortino]
sortino_max = sortino_list[idx_max_sortino]

# ---- Export to Excel ----
output_folder = r"C:\Users\anilk\OneDrive\Python"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "retirement_simulation.xlsx")

summary_df = pd.DataFrame({
    "Metric": ["Expected Return", "Volatility", "Sharpe Ratio", 
               "Sortino Ratio", "Max Drawdown"],
    "Current Portfolio": [port_return, port_vol, sharpe_ratio, sortino_ratio, max_drawdown],
    "Max Sharpe Portfolio": [ret_max_sharpe, vol_max_sharpe, sharpe_max, np.nan, np.nan],
    "Max Sortino Portfolio": [ret_max_sortino, vol_max_sortino, np.nan, sortino_max, np.nan]
})

df_weights = pd.DataFrame({
    "Asset": assets,
    "Current Weights": weights_current,
    "Max Sharpe Weights": w_max_sharpe,
    "Max Sortino Weights": w_max_sortino
})

with pd.ExcelWriter(output_file) as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    df_weights.to_excel(writer, sheet_name="Weights", index=False)
    df_percentiles.to_excel(writer, sheet_name="Percentiles", index=False)
    pd.DataFrame({"Final Balances": final_balances}).to_excel(writer, sheet_name="FinalBalances", index=False)

print(f"✅ Results exported to: {output_file}")

# ---- Efficient Frontier Plot ----
plt.figure(figsize=(10,6))
plt.scatter(vols_list, returns_list, c=sharpe_list, cmap="viridis", alpha=0.5)
plt.colorbar(label="Sharpe Ratio")
plt.scatter(port_vol, port_return, c="red", marker="*", s=200, label="Current Portfolio")
plt.scatter(vol_max_sharpe, ret_max_sharpe, c="gold", marker="*", s=200, label="Max Sharpe")
plt.scatter(vol_max_sortino, ret_max_sortino, c="green", marker="*", s=200, label="Max Sortino")
plt.title("Efficient Frontier (Simplified)")
plt.xlabel("Volatility (Std Dev)")
plt.ylabel("Expected Return")
plt.legend()
plt.grid(True)
plt.show()
