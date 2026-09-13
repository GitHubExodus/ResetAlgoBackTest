import numpy as np
import pandas as pd
import numba as nb


@nb.jit(nopython=True)
def run_backtest_kernel(
    top_100_mask: np.ndarray,
    close_prices: np.ndarray,
    fixed_weight: float = 0.01
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated portfolio accounting and trade tracking engine.
    
    Parameters:
    -----------
    top_100_mask : np.ndarray (shape: Num_Tickers, Timestamps)
        Boolean array where True indicates membership in top 100.
    close_prices : np.ndarray (shape: Num_Tickers, Timestamps)
        2D matrix of daily close prices.
    fixed_weight : float
        Contribution weight of each active stock per day (default 1% = 0.01).
        
    Returns:
    --------
    daily_returns : np.ndarray (shape: Timestamps)
        Daily portfolio return percentages.
    equity_curve : np.ndarray (shape: Timestamps)
        Normalized portfolio equity curve starting at 1.0 (or 100%).
    active_counts : np.ndarray (shape: Timestamps)
        Count of active positions per bar.
    """
    num_tickers, num_bars = top_100_mask.shape
    
    daily_returns = np.zeros(num_bars, dtype=np.float64)
    equity_curve = np.ones(num_bars, dtype=np.float64)
    active_counts = np.zeros(num_bars, dtype=np.float64)
    
    # Calculate daily individual asset return matrix: (P_t - P_{t-1}) / P_{t-1}
    asset_daily_returns = np.zeros((num_tickers, num_bars), dtype=np.float64)
    for t in range(num_tickers):
        for b in range(1, num_bars):
            prev_p = close_prices[t, b - 1]
            curr_p = close_prices[t, b]
            if prev_p > 0 and not np.isnan(prev_p) and not np.isnan(curr_p):
                asset_daily_returns[t, b] = (curr_p - prev_p) / prev_p

    # Compute daily weighted portfolio performance
    for b in range(1, num_bars):
        active_mask = top_100_mask[:, b - 1]  # Positions held entering bar b
        n_active = np.sum(active_mask)
        active_counts[b] = n_active
        
        if n_active > 0:
            # Sum returns of active positions scaled by fixed weight (1%)
            total_active_return = 0.0
            for t in range(num_tickers):
                if active_mask[t]:
                    total_active_return += asset_daily_returns[t, b]
            
            daily_returns[b] = total_active_return * fixed_weight
        
        equity_curve[b] = equity_curve[b - 1] * (1.0 + daily_returns[b])
        
    return daily_returns, equity_curve, active_counts


def extract_trade_log(
    top_100_mask: np.ndarray,
    close_prices: np.ndarray,
    timestamps: np.ndarray,
    tickers: list[str]
) -> pd.DataFrame:
    """
    Identifies entry (BUY) and exit (EXITS) timestamps for every stock 
    and calculates individual trade percentage returns.
    """
    num_tickers, num_bars = top_100_mask.shape
    trades = []

    for t in range(num_tickers):
        ticker = tickers[t] if tickers else f"STOCK_{t}"
        in_position = False
        entry_idx = 0
        entry_price = 0.0

        for b in range(num_bars):
            is_member = top_100_mask[t, b]

            # Entry condition: Stock enters top 100
            if is_member and not in_position:
                in_position = True
                entry_idx = b
                entry_price = close_prices[t, b]

            # Exit condition: Stock drops out of top 100 or timeline ends
            elif (not is_member and in_position) or (b == num_bars - 1 and in_position):
                in_position = False
                exit_idx = b
                exit_price = close_prices[t, b]

                if entry_price > 0:
                    trade_pct = ((exit_price - entry_price) / entry_price) * 100.0
                else:
                    trade_pct = 0.0

                trades.append({
                    "ticker": ticker,
                    "entry_time": timestamps[entry_idx],
                    "exit_time": timestamps[exit_idx],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "holding_period_days": exit_idx - entry_idx,
                    "trade_return_pct": trade_pct
                })

    return pd.DataFrame(trades)


class BacktestEngine:
    def __init__(self, fixed_weight: float = 0.01):
        self.fixed_weight = fixed_weight

    def run(
        self,
        top_100_mask: np.ndarray,
        close_prices: np.ndarray,
        timestamps: np.ndarray,
        tickers: list[str] = None
    ) -> dict:
        """
        Main interface executing portfolio accounting and generating output formats.
        """
        daily_returns, equity_curve, active_counts = run_backtest_kernel(
            top_100_mask, close_prices, self.fixed_weight
        )
        
        trade_log_df = extract_trade_log(
            top_100_mask, close_prices, timestamps, tickers
        )
        
        equity_df = pd.DataFrame({
            "timestamp": timestamps,
            "daily_return": daily_returns,
            "equity_curve": equity_curve,
            "active_positions": active_counts
        })

        return {
            "equity_df": equity_df,
            "trade_log": trade_log_df,
            "equity_curve_array": equity_curve,
            "daily_returns_array": daily_returns
        }