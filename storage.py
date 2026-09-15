import io
import json
import boto3
import numpy as np
import pandas as pd
from botocore.config import Config


class R2StorageManager:
    """Manages serialization and uploading of processed backtest and ranking outputs

    to Cloudflare R2 under the stats/ path.
    """

    def __init__(
        self,
        endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        bucket_name: str,
        stats_prefix: str = "stats/",
    ):
        self.bucket_name = bucket_name
        self.stats_prefix = stats_prefix.strip("/") + "/"
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=Config(signature_version="s3v4"),
        )

    def _upload_bytes(self, data_bytes: bytes, filename: str, content_type: str):
        """Helper to stream in-memory binary data to Cloudflare R2."""
        key = f"{self.stats_prefix}{filename}"
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data_bytes,
                ContentType=content_type,
            )
            print(f"[Upload Success] s3://{self.bucket_name}/{key}")
        except Exception as e:
            print(f"[Upload Error] Failed to upload {key}: {e}")
            raise e

    def save_valid_tickers(self, valid_tickers: list[str] | np.ndarray):
        """Serializes and uploads the complete-data master stock ticker list as JSON."""
        if isinstance(valid_tickers, np.ndarray):
            valid_tickers = valid_tickers.tolist()

        payload = json.dumps({"complete_tickers": valid_tickers}, indent=2).encode("utf-8")
        self._upload_bytes(payload, "valid_tickers.json", "application/json")

    def save_top_100_rankings(
        self,
        timestamps: list[str] | np.ndarray,
        valid_tickers: list[str] | np.ndarray,
        top_100_matrix: np.ndarray,
        max_rank_cols: int = 100,
    ):
        """Serializes and uploads daily top ordered stock lists per timestamp as CSV.

        Safely handles cases where available tickers < 100 by padding missing rank positions.
        """
        tickers_arr = np.array(valid_tickers)
        num_tickers = len(tickers_arr)
        records = []

        # Determine target rank positions count based on tickers or requested max
        actual_k = min(num_tickers, max_rank_cols)

        for t_idx, ts in enumerate(timestamps):
            row = {"timestamp": str(ts)}

            if top_100_matrix.dtype == bool:
                # Handle boolean flag matrix (Num_Tickers, Timestamps)
                active_indices = np.where(top_100_matrix[:, t_idx])[0]
                ranked_tickers = tickers_arr[active_indices]
            else:
                # Handle integer index matrix (Timestamps, Top_K)
                top_indices = top_100_matrix[t_idx]
                # Filter out invalid indices or indices beyond dataset bounds
                valid_indices = top_indices[top_indices < num_tickers]
                ranked_tickers = tickers_arr[valid_indices]

            # Populate present rank positions
            for rank_pos in range(1, max_rank_cols + 1):
                if rank_pos <= len(ranked_tickers):
                    row[f"rank_{rank_pos}"] = ranked_tickers[rank_pos - 1]
                else:
                    # Pad missing ranks with empty strings for small test sets
                    row[f"rank_{rank_pos}"] = ""

            records.append(row)

        df = pd.DataFrame(records)
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        self._upload_bytes(csv_buffer.getvalue(), "daily_top_100_rankings.csv", "text/csv")
    
    def save_individual_ranks(
        self,
        timestamps: list[str] | np.ndarray,
        valid_tickers: list[str] | np.ndarray,
        list1_ranks: np.ndarray,
        list2_ranks: np.ndarray,
        list3_ranks: np.ndarray,
    ):
        """Serializes and uploads detailed daily cross-sectional ranks for each stock."""
        tickers_arr = np.array(valid_tickers)
        ts_arr = np.array(timestamps)
        
        # Flatten panel data (Tickers x Timestamps) into tidy tabular structure
        num_tickers, num_days = list1_ranks.shape
        
        ticker_col = np.repeat(tickers_arr, num_days)
        ts_col = np.tile(ts_arr, num_tickers)
        
        df = pd.DataFrame({
            "ticker": ticker_col,
            "timestamp": ts_col,
            "rank_ema_avg": list1_ranks.ravel(),
            "rank_vol_20d": list2_ranks.ravel(),
            "rank_min_combined": list3_ranks.ravel(),
        })

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        self._upload_bytes(csv_buffer.getvalue(), "individual_list_ranks.csv", "text/csv")

    def save_portfolio_equity_curve(
        self,
        timestamps: list[str] | np.ndarray,
        equity_curve: np.ndarray,
        daily_contributions: np.ndarray | None = None,
    ):
        """Serializes and uploads the daily 1% normalized portfolio contribution equity curve."""
        df = pd.DataFrame({
            "timestamp": timestamps,
            "portfolio_equity": equity_curve,
        })
        
        if daily_contributions is not None:
            df["daily_contribution_pct"] = daily_contributions

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        self._upload_bytes(csv_buffer.getvalue(), "daily_portfolio_equity_curve.csv", "text/csv")


if __name__ == "__main__":
    # Example execution interface
    storage = R2StorageManager(
        endpoint_url="https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com",
        aws_access_key_id="00e18b0c16ecb3395cd6f7c8e0eb3554",
        aws_secret_access_key="33799355abaedc234309dbfbc80a2a66c3bfd856f0dcaecf0031e1fbcbcd84a0",
        bucket_name="stocks-data",
    )