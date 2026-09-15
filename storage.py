import io
import json
import boto3
import numpy as np
import pandas as pd
from botocore.config import Config


class R2StorageManager:
    """Manages serialization and uploading of processed trading metrics, ranks,

    top 100 constituents, and equity curves back to Cloudflare R2 under the stats/ folder.
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
        """Helper to stream in-memory binary/string data to Cloudflare R2."""
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

        payload = json.dumps({"valid_tickers": valid_tickers, "total_count": len(valid_tickers)}, indent=2).encode("utf-8")
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

        for t_idx, ts in enumerate(timestamps):
            row = {"timestamp": str(ts)}

            if top_100_matrix.dtype == bool:
                # Boolean mask matrix shape: (Num_Tickers, Timestamps)
                active_indices = np.where(top_100_matrix[:, t_idx])[0]
                ranked_tickers = tickers_arr[active_indices]
            else:
                # Integer index matrix shape: (Timestamps, Top_K)
                top_indices = top_100_matrix[t_idx]
                # Filter out invalid indices (e.g. -1 padding for small datasets) or out-of-bounds indices
                valid_indices = top_indices[(top_indices >= 0) & (top_indices < num_tickers)]
                ranked_tickers = tickers_arr[valid_indices]

            # Populate available rank positions and pad missing ones up to max_rank_cols
            for rank_pos in range(1, max_rank_cols + 1):
                if rank_pos <= len(ranked_tickers):
                    row[f"rank_{rank_pos}"] = ranked_tickers[rank_pos - 1]
                else:
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
        rank_matrix: np.ndarray,
    ):
        """Serializes individual stock ranks into a wide-format CSV table.

        Rows = Timestamps (ordered chronologically from oldest to newest)
        Columns = Tickers
        Values = Numerical rank on that specific timestamp
        """
        ts_list = [str(ts) for ts in timestamps]
        tickers_list = list(valid_tickers)

        # Ensure rank_matrix has shape (Timestamps, Tickers)
        if rank_matrix.shape == (len(tickers_list), len(ts_list)):
            rank_data = rank_matrix.T
        else:
            rank_data = rank_matrix

        # Construct wide-format DataFrame
        df = pd.DataFrame(rank_data, index=ts_list, columns=tickers_list)
        
        # Sort chronologically from oldest to newest
        df.index.name = "timestamp"
        df = df.sort_index(ascending=True).reset_index()

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        self._upload_bytes(csv_buffer.getvalue(), "individual_ranks_wide.csv", "text/csv")

    def save_portfolio_equity_curve(
        self,
        timestamps: list[str] | np.ndarray,
        equity_curve: np.ndarray,
        daily_contributions: np.ndarray | None = None,
    ):
        """Serializes and uploads the daily portfolio contribution equity curve."""
        df = pd.DataFrame({
            "timestamp": timestamps,
            "portfolio_equity": equity_curve,
        })
        
        if daily_contributions is not None:
            df["daily_contribution"] = daily_contributions

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        self._upload_bytes(csv_buffer.getvalue(), "daily_portfolio_equity_curve.csv", "text/csv")


if __name__ == "__main__":
    storage = R2StorageManager(
        endpoint_url="https://<account_id>.r2.cloudflarestorage.com",
        aws_access_key_id="<R2_ACCESS_KEY>",
        aws_secret_access_key="<R2_SECRET_KEY>",
        bucket_name="stock-data-bucket",
    )
    print("Chat 5 storage manager module initialized successfully.")