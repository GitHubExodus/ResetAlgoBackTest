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
    ):
        """Serializes and uploads daily top 100 ordered stock lists per timestamp as CSV."""
        tickers_arr = np.array(valid_tickers)
        records = []

        # Parse matrix (Timestamps x Top 100 Index Positions)
        for t_idx, ts in enumerate(timestamps):
            top_indices = top_100_matrix[t_idx]
            ranked_tickers = tickers_arr[top_indices]
            
            row = {"timestamp": str(ts)}
            for rank_pos, ticker in enumerate(ranked_tickers, start=1):
                row[f"rank_{rank_pos}"] = ticker
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