import numpy as np
import numba as nb


@nb.jit(nopython=True, fastmath=True)
def _rank_1d_descending(arr: np.ndarray) -> np.ndarray:
    """Computes ordinal 1-based ranks for a 1D float array in descending order.

    Higher values get lower rank numbers (1 = highest value).
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
    list1_matrix: np.ndarray, list2_matrix: np.ndarray, top_k: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes cross-sectional ranks bar-by-bar and extracts top K keepers.

    Parameters:
    -----------
    list1_matrix : np.ndarray
        2D array of shape (Num_Tickers, Timestamps) for EMA of average score.
    list2_matrix : np.ndarray
        2D array of shape (Num_Tickers, Timestamps) for 20-day rolling volume.
    top_k : int
        Number of top stocks to keep per bar.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - ranks_list1: (Num_Tickers, Timestamps) integer ranks
        - ranks_list2: (Num_Tickers, Timestamps) integer ranks
        - ranks_min: (Num_Tickers, Timestamps) minimum combined rank (List 3)
        - top100_flags: (Num_Tickers, Timestamps) boolean flags for top K keepers
        - top100_ordered_indices: (Timestamps, top_k) ticker indices ordered by min rank
    """
    num_tickers = list1_matrix.shape[0]
    num_bars = list1_matrix.shape[1]

    ranks_list1 = np.zeros((num_tickers, num_bars), dtype=np.int64)
    ranks_list2 = np.zeros((num_tickers, num_bars), dtype=np.int64)
    ranks_min = np.zeros((num_tickers, num_bars), dtype=np.int64)
    top100_flags = np.zeros((num_tickers, num_bars), dtype=np.bool_)
    top100_ordered_indices = np.zeros((num_bars, top_k), dtype=np.int64)

    for b in range(num_bars):
        # Extract 1D cross-section for the current bar
        col1 = list1_matrix[:, b]
        col2 = list2_matrix[:, b]

        # Compute ordinal ranks (1 = highest value)
        r1 = _rank_1d_descending(col1)
        r2 = _rank_1d_descending(col2)

        ranks_list1[:, b] = r1
        ranks_list2[:, b] = r2

        # List 3: Minimum rank between List 1 and List 2
        bar_min_ranks = np.minimum(r1, r2)
        ranks_min[:, b] = bar_min_ranks

        # Extract top K indices by sorting the bar's minimum ranks ascending
        sorted_ticker_indices = np.argsort(bar_min_ranks)
        top_k_indices = sorted_ticker_indices[:top_k]

        top100_ordered_indices[b, :] = top_k_indices
        top100_flags[top_k_indices, b] = True

    return (
        ranks_list1,
        ranks_list2,
        ranks_min,
        top100_flags,
        top100_ordered_indices,
    )


def compute_rankings_and_top100(
    list1_matrix: np.ndarray, list2_matrix: np.ndarray, top_k: int = 100
) -> dict[str, np.ndarray]:
    """Computes cross-sectional rankings directly from two positional matrix inputs.

    Returns dictionary mapping containing aliases for both top_100_matrix and top100_flags.
    """
    # Ensure C-contiguous float64 numpy matrices for Numba compatibility
    l1 = np.ascontiguousarray(list1_matrix, dtype=np.float64)
    l2 = np.ascontiguousarray(list2_matrix, dtype=np.float64)

    (
        ranks_list1,
        ranks_list2,
        ranks_min,
        top100_flags,
        top100_ordered_indices,
    ) = compute_cross_sectional_ranks_numba(l1, l2, top_k=top_k)

    return {
        "ranks_list1": ranks_list1,
        "ranks_list2": ranks_list2,
        "ranks_min": ranks_min,
        "top100_flags": top100_flags,
        "top_100_matrix": top100_flags,
        "top100_matrix": top100_flags,
        "top100_ordered_indices": top100_ordered_indices,
        "list1_ranks": ranks_list1,
        "list2_ranks": ranks_list2,
        "list3_ranks": ranks_min,
    }


def process_cross_sectional_rankings(
    indicator_outputs: dict[str, np.ndarray], top_k: int = 100
) -> dict[str, np.ndarray]:
    """Legacy dictionary-wrapper maintained for backward compatibility."""
    return compute_rankings_and_top100(
        indicator_outputs["avg_ema"],
        indicator_outputs["vol_sma"],
        top_k=top_k,
    )


if __name__ == "__main__":
    num_tickers = 1000
    num_bars = 250

    np.random.seed(42)
    mock_avg_ema = np.random.randn(num_tickers, num_bars)
    mock_vol_sma = np.random.rand(num_tickers, num_bars) * 1000000.0

    # Test direct positional call matching main.py signature
    results = compute_rankings_and_top100(mock_avg_ema, mock_vol_sma, top_k=100)

    print("Ranking processing complete.")
    print(f"Ranks Min Shape: {results['ranks_min'].shape}")
    print(f"Top 100 Matrix Shape: {results['top_100_matrix'].shape}")
    print(
        f"Top 100 Ordered Indices Shape: {results['top100_ordered_indices'].shape}"
    )