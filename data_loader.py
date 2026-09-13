import io
import random
import boto3
import numpy as np
import pandas as pd
from botocore.config import Config


class R2DataLoader:

    def __init__(
        self,
        endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        bucket_name: str,
        sample_size: int = 1000,
    ):
        self.bucket_name = bucket_name
        self.sample_size = sample_size
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=Config(signature_version="s3v4"),
        )

    def list_stock_csvs(self) -> list[str]:
        """Scans the flat R2 bucket for CSV files (excluding folders)."""
        csv_files = []
        paginator = self.s3_client.get_paginator("list_objects_v2")

        try:
            for page in paginator.paginate(Bucket=self.bucket_name):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        # Select only top-level CSV files
                        if "/" not in key and key.endswith(".csv"):
                            csv_files.append(key)
        except Exception as e:
            print(f"Error fetching bucket keys: {e}")
            raise

        return csv_files

    def fetch_and_resample(self, key: str) -> pd.DataFrame | None:
        """Downloads a single 1-minute CSV from R2 and resamples it to daily bars."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=key
            )
            data_bytes = response["Body"].read()

            # Load 1-minute CSV data into pandas for initial time handling
            df = pd.read_csv(io.BytesIO(data_bytes))

            # Ensure timestamp column is parsed
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            else:
                df.index = pd.to_datetime(df.index)

            # Resample 1-minute OHLCV data into Daily bars
            daily_df = (
                df.resample("1D")
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna(subset=["close"])
            )

            return daily_df

        except Exception as e:
            print(f"Failed to process key {key}: {e}")
            return None

    def process_dataset(self) -> dict:
        """Main execution loop to sample keys, fetch data, align into a unified NumPy panel,

        and filter out tickers with missing timeline history.
        """
        all_keys = self.list_stock_csvs()

        if len(all_keys) > self.sample_size:
            selected_keys = random.sample(all_keys, self.sample_size)
        else:
            selected_keys = all_keys

        raw_data = {}
        for key in selected_keys:
            ticker = key.replace(".csv", "")
            df = self.fetch_and_resample(key)
            if df is not None and not df.empty:
                raw_data[ticker] = df

        if not raw_data:
            raise ValueError("No valid daily data was downloaded or resampled.")

        # 1. Construct unified timeline across all fetched tickers
        all_timestamps = sorted(
            list(
                set(
                    ts
                    for df in raw_data.values()
                    for ts in df.index.strftime("%Y-%m-%d")
                )
            )
        )
        timeline_len = len(all_timestamps)
        tickers = np.array(list(raw_data.keys()))
        num_tickers = len(tickers)

        # 2. Allocate 3D NumPy panel: (Tickers, Timestamps, OHLCV [5])
        # Data ordering: 0: open, 1: high, 2: low, 3: close, 4: volume
        data_panel = np.full(
            (num_tickers, timeline_len, 5), np.nan, dtype=np.float64
        )
        date_map = {date: idx for idx, date in enumerate(all_timestamps)}

        for t_idx, ticker in enumerate(tickers):
            df = raw_data[ticker]
            for ts, row in df.iterrows():
                date_str = ts.strftime("%Y-%m-%d")
                if date_str in date_map:
                    d_idx = date_map[date_str]
                    data_panel[t_idx, d_idx, 0] = row["open"]
                    data_panel[t_idx, d_idx, 1] = row["high"]
                    data_panel[t_idx, d_idx, 2] = row["low"]
                    data_panel[t_idx, d_idx, 3] = row["close"]
                    data_panel[t_idx, d_idx, 4] = row["volume"]

        # 3. Validation filter using NumPy: Check for complete timeline history
        # True if no NaNs are found across any daily bar for a stock
        complete_mask = ~np.isnan(data_panel).any(axis=(1, 2))

        valid_tickers = tickers[complete_mask].tolist()
        valid_panel = data_panel[complete_mask]

        return {
            "timestamps": np.array(all_timestamps),
            "valid_tickers": valid_tickers,
            "panel": valid_panel,  # Cleaned NumPy panel ready for indicators
        }


if __name__ == "__main__":
    # Test execution placeholder credentials
    loader = R2DataLoader(
        endpoint_url="https://98f8e959e677f16bddcf44f609fec6a0.r2.cloudflarestorage.com",
        aws_access_key_id="00e18b0c16ecb3395cd6f7c8e0eb3554",
        aws_secret_access_key="33799355abaedc234309dbfbc80a2a66c3bfd856f0dcaecf0031e1fbcbcd84a0",
        bucket_name="stocks-data",
        sample_size=1,
    )

    dataset = loader.process_dataset()
    print(f"Complete Tickers Count: {len(dataset['valid_tickers'])}")
    print(f"Panel Shape: {dataset['panel'].shape}")