import numpy as np
import numba as nb


@nb.jit(nopython=True)
def calculate_ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Calculates Exponential Moving Average over a 1D array using Numba."""
    n = len(arr)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema

    alpha = 2.0 / (period + 1.0)
    ema[0] = arr[0]

    for i in range(1, n):
        if np.isnan(arr[i]):
            ema[i] = ema[i - 1]
        else:
            ema[i] = (arr[i] * alpha) + (ema[i - 1] * (1.0 - alpha))

    return ema


@nb.jit(nopython=True)
def calculate_rolling_sma(arr: np.ndarray, window: int) -> np.ndarray:
    """Calculates a simple rolling moving average over a 1D array using Numba."""
    n = len(arr)
    sma = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return sma

    current_sum = 0.0
    for i in range(window):
        current_sum += arr[i]
    sma[window - 1] = current_sum / window

    for i in range(window, n):
        current_sum += arr[i] - arr[i - window]
        sma[i] = current_sum / window

    return sma


@nb.jit(nopython=True)
def calculate_single_stock_indicators(
    open_prices: np.ndarray,
    close_prices: np.ndarray,
    volumes: np.ndarray,
    years: np.ndarray,
    months: np.ndarray,
    weeks: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Calculates YTD%, MTD%, WTD%, Daily%, average score, 14 EMA, and 20 Vol SMA

    for a single stock using Numba acceleration.
    """
    n = len(close_prices)

    ytd_pct = np.zeros(n, dtype=np.float64)
    mtd_pct = np.zeros(n, dtype=np.float64)
    wtd_pct = np.zeros(n, dtype=np.float64)
    day_pct = np.zeros(n, dtype=np.float64)
    bar_avg = np.zeros(n, dtype=np.float64)

    ytd_base = open_prices[0]
    mtd_base = open_prices[0]
    wtd_base = open_prices[0]

    for i in range(n):
        if i > 0:
            # Baseline resets on date changes (Pine Script close[1] logic)
            if years[i] != years[i - 1]:
                ytd_base = close_prices[i - 1]
            if months[i] != months[i - 1]:
                mtd_base = close_prices[i - 1]
            if weeks[i] != weeks[i - 1]:
                wtd_base = close_prices[i - 1]

        # Calculate percentage changes
        ytd_pct[i] = (
            ((close_prices[i] - ytd_base) / ytd_base) * 100.0
            if ytd_base != 0.0
            else 0.0
        )
        mtd_pct[i] = (
            ((close_prices[i] - mtd_base) / mtd_base) * 100.0
            if mtd_base != 0.0
            else 0.0
        )
        wtd_pct[i] = (
            ((close_prices[i] - wtd_base) / wtd_base) * 100.0
            if wtd_base != 0.0
            else 0.0
        )

        if open_prices[i] != 0.0:
            day_pct[i] = (
                (close_prices[i] - open_prices[i]) / open_prices[i]
            ) * 100.0
        else:
            day_pct[i] = 0.0

        bar_avg[i] = (ytd_pct[i] + mtd_pct[i] + wtd_pct[i] + day_pct[i]) / 4.0

    avg_ema = calculate_ema(bar_avg, 14)
    vol_sma = calculate_rolling_sma(volumes, 20)

    return ytd_pct, mtd_pct, wtd_pct, day_pct, bar_avg, avg_ema, vol_sma


def compute_panel_indicators(
    panel: np.ndarray, dates: np.ndarray
) -> dict[str, np.ndarray]:
    """Processes a 3D dataset panel (Num_Tickers, Timestamps, OHLCV) across all stocks.

    Parameters:
    -----------
    panel : np.ndarray
        Shape (Num_Tickers, Timestamps, 5) where last axis is [Open, High, Low, Close, Volume]
    dates : np.ndarray
        Array of timestamp strings or datetime objects matching the length of Timestamps.
    """
    dt_series = (
        dates if isinstance(dates, np.ndarray) else np.array(dates)
    ).astype("datetime64[D]")
    years = dt_series.astype("datetime64[Y]").astype(int) + 1970
    months = dt_series.astype("datetime64[M]").astype(int) % 12 + 1
    weeks = (dt_series.astype("datetime64[W]").astype(int) + 3) % 52 + 1

    num_tickers, num_bars, _ = panel.shape

    avg_ema_matrix = np.zeros((num_tickers, num_bars), dtype=np.float64)
    vol_sma_matrix = np.zeros((num_tickers, num_bars), dtype=np.float64)
    bar_avg_matrix = np.zeros((num_tickers, num_bars), dtype=np.float64)

    for t in range(num_tickers):
        open_p = panel[t, :, 0]
        close_p = panel[t, :, 3]
        volume = panel[t, :, 4]

        _, _, _, _, bar_avg, avg_ema, vol_sma = (
            calculate_single_stock_indicators(
                open_p, close_p, volume, years, months, weeks
            )
        )

        bar_avg_matrix[t] = bar_avg
        avg_ema_matrix[t] = avg_ema
        vol_sma_matrix[t] = vol_sma

    return {
        "avg_ema": avg_ema_matrix,  # List 1 input
        "vol_sma": vol_sma_matrix,  # List 2 input
        "bar_avg": bar_avg_matrix,
    }