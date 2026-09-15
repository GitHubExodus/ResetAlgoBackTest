import numpy as np
import numba as nb


@nb.jit(nopython=True, fastmath=True)
def _rank_1d_descending(arr: np.ndarray) -> np.ndarray:
    """Computes ordinal 1-based ranks for a 1D float array in descending order.

    Higher values get lower rank numbers (1 = highest score).
    NaNs are assigned a high rank penalty (placed last).
    """
    n = len(arr)
    ranks = np.zeros(n, dtype=np.int64)

    # Temporary storage for non-NaN values and their original indices
    valid_indices = np.empty(n, dtype=np.int64)
    valid_values = np.empty(n, dtype=np.float64)
    valid_count = 0

    for i in range(n):
        val = arr[i]
        if not np.isnan(val):
            valid_indices[valid_count] = i
            valid_values[valid_count] = val
            valid_count += 1
        else:
            ranks[i] = n  # High penalty rank for missing data

    if valid_count > 0:
        # Sort indices of valid entries in descending order of their values
        sorted_order = np.argsort(valid_values[:valid_count])[::-1]
        for rank_idx in range(valid_count):
            orig_idx = valid_indices[sorted_order[rank_idx]]
            ranks[orig_idx] = rank_idx + 1  # 1-based rank

    return ranks


@nb.jit(nopython=True, parallel=False)
def compute_cross_sectional_ranks_numba(
    ema_matrix: np.ndarray, top_k: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes single-metric cross-sectional ranks bar-by-bar using the EMA score matrix.

    Parameters:
    -----------
    ema_matrix : np.ndarray
        2D array of shape (Num_Tickers, Timestamps) representing the EMA score.
    top_k : int
        Number of top stocks to keep per bar.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - ranks_ema: (Num_Tickers, Timestamps) integer ranks (1 = highest score)
        - top100_flags: (Num_Tickers, Timestamps) boolean flags for top K keepers
        - top100_ordered_indices: (Timestamps, top_k) ticker indices ordered by rank
    """
    num_tickers = ema_matrix.shape[0]
    num_bars = ema_matrix.shape[1]

    # Dynamically bound top_k to available tickers for small test batches
    effective_k = min(top_k, num_tickers)

    ranks_ema = np.zeros((num_tickers, num_bars), dtype=np.int64)
    top100_flags = np.zeros((num_tickers, num_bars), dtype=np.bool_)
    
    # Pre-fill with -1 as padding for unused rank slots when effective_k < top_k
    top100_ordered_indices = np.full((num_bars, top_k), -1, dtype=np.int64)

    for b in range(num_bars):
        # Extract 1D cross-section for the current bar
        col_ema = ema_matrix[:, b]

        # Compute ordinal ranks (1 = highest EMA score)
        r_ema = _rank_1d_descending(col_ema)
        ranks_ema[:, b] = r_ema

        # Extract top K indices by sorting the bar's EMA ranks ascending (1 is best)
        sorted_ticker_indices = np.argsort(r_ema)
        top_k_indices = sorted_ticker_indices[:effective_k]

        top100_ordered_indices[b, :effective_k] = top_k_indices
        top100_flags[top_k_indices, b] = True

    return (
        ranks_ema,
        top100_flags,
        top100_ordered_indices,
    )


def compute_rankings_and_top100(
    *args, top_k: int = 100, **kwargs
) -> dict[str, np.ndarray]:
    """Flexible wrapper extracting cross-sectional rankings based exclusively on the EMA matrix."""
    # Unpack EMA matrix regardless of input positional/keyword signature
    if len(args) >= 1:
        ema_matrix = args[0]
    elif "ema_matrix" in kwargs:
        ema_matrix = kwargs["ema_matrix"]
    elif "list1_matrix" in kwargs:
        ema_matrix = kwargs["list1_matrix"]
    elif isinstance(args, tuple) and len(args) == 0 and "indicator_outputs" in kwargs:
        ema_matrix = kwargs["indicator_outputs"]["avg_ema"]
    else:
        raise ValueError("Could not extract EMA score matrix from function arguments.")

    # Cast to C-contiguous 2D float64 matrix for Numba compiler compatibility
    ema_matrix = np.ascontiguousarray(ema_matrix, dtype=np.float64)

    (
        ranks_ema,
        top100_flags,
        top100_ordered_indices,
    ) = compute_cross_sectional_ranks_numba(ema_matrix, top_k=top_k)

    return {
        "top_100_matrix": top100_flags,
        "top100_flags": top100_flags,
        "top100_matrix": top100_flags,
        "ranks_ema": ranks_ema,
        "ranks_list1": ranks_ema,
        "top100_ordered_indices": top100_ordered_indices,
    }


def process_cross_sectional_rankings(
    indicator_outputs: dict[str, np.ndarray], top_k: int = 100
) -> dict[str, np.ndarray]:
    """Wrapper function executing single-metric ranking on indicator dictionaries."""
    ema_matrix = indicator_outputs["avg_ema"]
    return compute_rankings_and_top100(ema_matrix, top_k=top_k)


if __name__ == "__main__":
    # Test execution placeholder
    num_tickers = 1000
    num_bars = 250

    np.random.seed(42)
    mock_avg_ema = np.random.randn(num_tickers, num_bars)

    results = compute_rankings_and_top100(mock_avg_ema, top_k=100)

    print("Single-metric ranking complete.")
    print(f"Top 100 Matrix Shape: {results['top_100_matrix'].shape}")
    print(f"EMA Ranks Shape: {results['ranks_ema'].shape}")
    print(f"Ordered Indices Shape: {results['top100_ordered_indices'].shape}")