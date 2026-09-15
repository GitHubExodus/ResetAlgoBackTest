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
        min_valid_ratio: float = 0.3,
    ):
        """Data Loader for Cloudflare R2 stock Parquet files.

        Parameters:
        -----------
        sample_size : int
            Number of stock Parquet files to randomly sample from the bucket.
        min_valid_ratio : float
            Minimum fraction of total timeline dates a stock must have data for
            (prior to ffill/bfill) to be retained in the final array.
        """
        self.bucket_name = bucket_name
        self.sample_size = sample_size
        self.min_valid_ratio = min_valid_ratio
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=Config(signature_version="s3v4"),
        )

    def list_stock_parquets(self) -> list[str]:
        """Scans the Cloudflare R2 bucket for Parquet files."""
        parquet_files = []
        paginator = self.s3_client.get_paginator("list_objects_v2")

        try:
            for page in paginator.paginate(Bucket=self.bucket_name):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if key.endswith(".parquet") or ".parquet" in key:
                            parquet_files.append(key)
        except Exception as e:
            print(f"Error fetching bucket keys: {e}")
            raise

        return parquet_files

    def fetch_and_resample(self, key: str) -> pd.DataFrame | None:
        """Downloads a single Parquet file from R2 using pd.read_parquet

        and resamples 1-minute OHLCV data into clean daily bars.
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=key
            )
            data_bytes = response["Body"].read()

            # Read parquet stream into Pandas DataFrame
            df = pd.read_parquet(io.BytesIO(data_bytes))

            # Standardize column headers to lowercase
            df.columns = df.columns.str.lower()

            # Identify timestamp column or use index
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            else:
                df.index = pd.to_datetime(df.index)

            # Sort chronological history
            df.sort_index(inplace=True)

            # Resample 1-minute OHLCV data into daily bars
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

            return daily_df if not daily_df.empty else None

        except Exception as e:
            print(f"Failed to process key {key}: {e}")
            return None

    def process_dataset(self) -> dict:
        """Downloads, resamples, aligns, imputes missing data, and builds a robust

        3D NumPy array of shape (Num_Tickers, Timestamps, 5).
        """
        all_keys = self.list_stock_parquets()

        if not all_keys:
            raise ValueError(
                f"No Parquet files found in bucket '{self.bucket_name}'."
            )

        if len(all_keys) > self.sample_size:
            selected_keys = random.sample(all_keys, self.sample_size)
        else:
            selected_keys = all_keys

        raw_data = {}
        for key in selected_keys:
            ticker = key.split("/")[-1].replace(".parquet", "")
            df = self.fetch_and_resample(key)
            if df is not None:
                raw_data[ticker] = df

        if not raw_data:
            raise ValueError(
                "No valid daily data was parsed from the downloaded Parquet files."
            )

        # 1. Establish master daily date index across all fetched stocks
        all_dates = sorted(
            list(
                set(
                    ts
                    for df in raw_data.values()
                    for ts in df.index.tz_localize(None)
                )
            )
        )
        master_index = pd.DatetimeIndex(all_dates)
        timeline_len = len(master_index)

        aligned_dfs = []
        valid_tickers = []

        # 2. Align each ticker to master index & apply forward-fill / backward-fill
        for ticker, df in raw_data.items():
            # Strip timezone if present to align smoothly
            df.index = df.index.tz_localize(None)

            # Check raw coverage ratio prior to imputation
            overlap_count = df.index.isin(master_index).sum()
            if (overlap_count / timeline_len) < self.min_valid_ratio:
                continue

            # Reindex to master timeline
            aligned = df.reindex(master_index)

            # Forward-fill gaps (weekend/holiday gaps or non-trading days)
            # Backward-fill early missing history (IPOs after dataset start)
            aligned[["open", "high", "low", "close"]] = aligned[
                ["open", "high", "low", "close"]
            ].ffill().bfill()

            # Fill volume missing bars with zero
            aligned["volume"] = aligned["volume"].fillna(0.0)

            aligned_dfs.append(
                aligned[["open", "high", "low", "close", "volume"]].values
            )
            valid_tickers.append(ticker)

        if not aligned_dfs:
            raise ValueError(
                "No tickers met the minimum timeline coverage criteria."
            )

        # 3. Stack into unified 3D NumPy panel: (Num_Tickers, Timestamps, 5)
        # Feature order -> 0: Open, 1: High, 2: Low, 3: Close, 4: Volume
        data_panel = np.stack(aligned_dfs, axis=0)

        # Final safety cleanup for any leftover NaNs
        if np.isnan(data_panel).any():
            data_panel = np.nan_to_num(data_panel, nan=0.0)

        return {
            "timestamps": master_index.strftime("%Y-%m-%d").to_numpy(),
            "valid_tickers": np.array(valid_tickers),
            "panel": data_panel,
        }


if __name__ == "__main__":
    loader = R2DataLoader(
        endpoint_url="https://<account_id>.r2.cloudflarestorage.com",
        aws_access_key_id="<R2_ACCESS_KEY>",
        aws_secret_access_key="<R2_SECRET_KEY>",
        bucket_name="stocks-data",
        sample_size=1000,
        min_valid_ratio=0.3,
    )

    # dataset = loader.process_dataset()
    # print(f"Valid Tickers: {len(dataset['valid_tickers'])}")
    # print(f"NumPy Panel Shape: {dataset['panel'].shape}")