from dagster import ConfigurableIOManager, InputContext, OutputContext
from typing import Optional, Union
import pandas as pd
import s3fs
import os


def get_asset_path_info(context: Union[InputContext, OutputContext]) -> tuple[str, Optional[str]]:
    """
    Extract asset path and partition key from context.
    
    Returns:
        tuple: (asset_path, partition_key)
    """
    asset_path = "/".join(context.asset_key.path)
    
    if context.has_asset_partitions:
        try:
            partition_key = context.asset_partition_key
        except Exception as e:
            partition_key = None
    elif context.has_partition_key:
        partition_key = context.partition_key
    else:
        partition_key = None
    
    return asset_path, partition_key


def build_file_path(asset_path: str, partition_key: Optional[str], extension: str, base_path: Optional[str] = None) -> str:
    """
    Build file path based on asset path, partition key, and extension.
    
    Args:
        asset_path: The asset path (e.g., "trades/raw_trades")
        partition_key: The partition key (e.g., "2024-01-15") or None
        extension: File extension (e.g., "csv", "parquet")
        base_path: Base path for local files (optional)
    
    Returns:
        str: Complete file path
    """
    if partition_key:
        file_name = f"{partition_key}.{extension}"
    else:
        file_name = f"{asset_path}.{extension}"
    
    if base_path:
        # Local file path
        return os.path.join(base_path, asset_path, file_name)
    else:
        # S3 path (no base_path)
        return f"{asset_path}/{file_name}"


class LocalCSVIOManager(ConfigurableIOManager):
    base_path: str = "./local_data"

    def _get_path(self, context: Union[InputContext, OutputContext]) -> Optional[str]:
        asset_path, partition_key = get_asset_path_info(context)
        if context.has_asset_partitions or context.has_partition_key:
            if partition_key is None:
                context.log.debug("Partition key is None")
                return None
        
        full_path = build_file_path(asset_path, partition_key, "csv", self.base_path)
        context.log.debug(f"Resolved path: {full_path}")
        return full_path

    def handle_output(self, context: OutputContext, obj: pd.DataFrame):
        path = self._get_path(context)
        if path is None:
            raise ValueError("Could not resolve output path")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        obj.to_csv(path, index=False)
        context.log.info(f"Saved DataFrame to {path}")

    def load_input(self, context: InputContext) -> Optional[pd.DataFrame]:
        path = self._get_path(context)
        if path is None:
            context.log.info("[LocalCSVIOManager] No resolved input partition — returning None.")
            return None

        if not os.path.exists(path):
            context.log.info(f"[LocalCSVIOManager] File does not exist: {path} — returning None.")
            return None

        context.log.info(f"[LocalCSVIOManager] Loading from {path}")
        return pd.read_csv(path)


class S3CSVIOManager(ConfigurableIOManager):
    s3_bucket: str
    s3_prefix: str = "analytics"

    def _get_path(self, context: Union[InputContext, OutputContext]) -> Optional[str]:
        asset_path, partition_key = get_asset_path_info(context)
        
        if context.has_partition_key and partition_key is None:
            return None
        
        s3_path = build_file_path(asset_path, partition_key, "csv")
        return f"{self.s3_bucket}/{self.s3_prefix}/{s3_path}"

    def handle_output(self, context: OutputContext, obj: pd.DataFrame):
        path = self._get_path(context)
        if path is None:
            raise ValueError("Could not resolve output path")

        fs = s3fs.S3FileSystem()
        with fs.open(f"s3://{path}", "w") as f:
            obj.to_csv(f, index=False)
        context.log.info(f"Saved DataFrame to s3://{path}")

    def load_input(self, context: InputContext) -> Optional[pd.DataFrame]:
        path = self._get_path(context)
        if path is None:
            context.log.info("[S3CSVIOManager] No resolved input partition — returning None.")
            return None

        fs = s3fs.S3FileSystem()
        try:
            with fs.open(f"s3://{path}", "r") as f:
                context.log.info(f"[S3CSVIOManager] Loading from s3://{path}")
                return pd.read_csv(f)
        except FileNotFoundError:
            context.log.info(f"[S3CSVIOManager] File does not exist: s3://{path} — returning None.")
            return None


class S3ParquetIOManager(ConfigurableIOManager):
    s3_bucket: str
    s3_prefix: str = "analytics"

    def _get_path(self, context: Union[InputContext, OutputContext]) -> Optional[str]:
        asset_path, partition_key = get_asset_path_info(context)
        
        if context.has_partition_key and partition_key is None:
            return None
        
        s3_path = build_file_path(asset_path, partition_key, "parquet")
        return f"{self.s3_bucket}/{self.s3_prefix}/{s3_path}"

    def handle_output(self, context: OutputContext, obj: pd.DataFrame):
        path = self._get_path(context)
        if path is None:
            raise ValueError("Could not resolve output path")

        fs = s3fs.S3FileSystem()
        with fs.open(f"s3://{path}", "wb") as f:
            obj.to_parquet(f, index=False)
        context.log.info(f"Saved DataFrame to s3://{path}")

    def load_input(self, context: InputContext) -> Optional[pd.DataFrame]:
        path = self._get_path(context)
        if path is None:
            context.log.info("[S3ParquetIOManager] No resolved input partition — returning None.")
            return None

        fs = s3fs.S3FileSystem()
        try:
            with fs.open(f"s3://{path}", "rb") as f:
                context.log.info(f"[S3ParquetIOManager] Loading from s3://{path}")
                return pd.read_parquet(f)
        except FileNotFoundError:
            context.log.info(f"[S3ParquetIOManager] File does not exist: s3://{path} — returning None.")
            return None