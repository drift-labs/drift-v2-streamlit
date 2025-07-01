from dagster import ConfigurableIOManager, InputContext, OutputContext
from typing import Optional, Union
import pandas as pd
import s3fs
import os


def _get_partition_key(context: Union[InputContext, OutputContext]) -> Optional[str]:
    """Extract partition key with consistent logic across all contexts."""
    if isinstance(context, OutputContext):
        return context.partition_key
    elif isinstance(context, InputContext):
        # For InputContext, we need to handle partition mapping correctly
        # Check if there are any asset partition keys available
        if hasattr(context, "asset_partition_keys"):
            partition_keys = list(context.asset_partition_keys)
            context.log.debug(f"Available asset_partition_keys: {partition_keys}")

            if len(partition_keys) == 0:
                context.log.debug("No asset partition keys available")
                return None
            elif len(partition_keys) == 1:
                partition_key = partition_keys[0]
                context.log.debug(f"Using single partition key: {partition_key}")
                return partition_key
            else:
                # Multiple partitions - use the first one or handle as needed
                partition_key = partition_keys[0]
                context.log.debug(
                    f"Multiple partition keys available, using first: {partition_key}"
                )
                return partition_key
        else:
            # Fallback - this might still fail if no partitions are available
            try:
                partition_key = context.asset_partition_key
                context.log.debug(f"Using fallback asset_partition_key: {partition_key}")
                return partition_key
            except Exception as e:
                context.log.debug(f"Failed to get partition key: {e}")
                return None
    else:
        raise ValueError("Unexpected context type")


class LocalCSVIOManager(ConfigurableIOManager):
    base_path: str = "./local_data"

    def _get_path(self, context: Union[InputContext, OutputContext]) -> Optional[str]:
        asset_path = "/".join(context.asset_key.path)

        if context.has_partition_key:
            partition_key = _get_partition_key(context)
            if partition_key is None:
                context.log.debug("Partition key is None")
                return None
            file_name = f"{partition_key}.csv"
        else:
            file_name = f"{asset_path}.csv"

        full_path = os.path.join(self.base_path, asset_path, file_name)
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
        # Handle the case where there are no input partitions available
        if hasattr(context, "asset_partition_keys"):
            partition_keys = list(context.asset_partition_keys)
            if len(partition_keys) == 0:
                context.log.info(
                    "[LocalCSVIOManager] No input partitions available — returning None."
                )
                return None

        path = self._get_path(context)
        if path is None:
            context.log.info(
                "[LocalCSVIOManager] No resolved input partition — returning None."
            )
            return None

        if not os.path.exists(path):
            context.log.info(
                f"[LocalCSVIOManager] File does not exist: {path} — returning None."
            )
            return None

        context.log.info(f"[LocalCSVIOManager] Loading from {path}")
        return pd.read_csv(path)


class S3CSVIOManager(ConfigurableIOManager):
    s3_bucket: str
    s3_prefix: str = "analytics"

    def _get_path(self, context: Union[InputContext, OutputContext]) -> Optional[str]:
        asset_path = "/".join(context.asset_key.path)

        if context.has_partition_key:
            partition_key = _get_partition_key(context)
            if partition_key is None:
                return None
            file_name = f"{partition_key}.csv"
        else:
            file_name = f"{asset_path}.csv"

        return f"{self.s3_bucket}/{self.s3_prefix}/{asset_path}/{file_name}"

    def handle_output(self, context: OutputContext, obj: pd.DataFrame):
        path = self._get_path(context)
        if path is None:
            raise ValueError("Could not resolve output path")
        
        fs = s3fs.S3FileSystem()
        with fs.open(f"s3://{path}", "w") as f:
            obj.to_csv(f, index=False)
        context.log.info(f"Saved DataFrame to s3://{path}")

    def load_input(self, context: InputContext) -> Optional[pd.DataFrame]:
        # Handle the case where there are no input partitions available
        if hasattr(context, "asset_partition_keys"):
            partition_keys = list(context.asset_partition_keys)
            if len(partition_keys) == 0:
                context.log.info(
                    "[S3CSVIOManager] No input partitions available — returning None."
                )
                return None

        path = self._get_path(context)
        if path is None:
            context.log.info(
                "[S3CSVIOManager] No resolved input partition — returning None."
            )
            return None

        fs = s3fs.S3FileSystem()
        try:
            with fs.open(f"s3://{path}", "r") as f:
                context.log.info(f"[S3CSVIOManager] Loading from s3://{path}")
                return pd.read_csv(f)
        except FileNotFoundError:
            context.log.info(
                f"[S3CSVIOManager] File does not exist: s3://{path} — returning None."
            )
            return None


class S3ParquetIOManager(ConfigurableIOManager):
    s3_bucket: str
    s3_prefix: str = "analytics"

    def _get_path(self, context: Union[InputContext, OutputContext]) -> Optional[str]:
        asset_path = "/".join(context.asset_key.path)

        if context.has_partition_key:
            partition_key = _get_partition_key(context)
            if partition_key is None:
                return None
            file_name = f"{partition_key}.parquet"
        else:
            file_name = f"{asset_path}.parquet"

        return f"{self.s3_bucket}/{self.s3_prefix}/{asset_path}/{file_name}"

    def handle_output(self, context: OutputContext, obj: pd.DataFrame):
        path = self._get_path(context)
        if path is None:
            raise ValueError("Could not resolve output path")
        
        fs = s3fs.S3FileSystem()
        with fs.open(f"s3://{path}", "wb") as f:
            obj.to_parquet(f, index=False)
        context.log.info(f"Saved DataFrame to s3://{path}")

    def load_input(self, context: InputContext) -> Optional[pd.DataFrame]:
        # Handle the case where there are no input partitions available
        if hasattr(context, "asset_partition_keys"):
            partition_keys = list(context.asset_partition_keys)
            if len(partition_keys) == 0:
                context.log.info(
                    "[S3ParquetIOManager] No input partitions available — returning None."
                )
                return None

        path = self._get_path(context)
        if path is None:
            context.log.info(
                "[S3ParquetIOManager] No resolved input partition — returning None."
            )
            return None

        fs = s3fs.S3FileSystem()
        try:
            with fs.open(f"s3://{path}", "rb") as f:
                context.log.info(f"[S3ParquetIOManager] Loading from s3://{path}")
                return pd.read_parquet(f)
        except FileNotFoundError:
            context.log.info(
                f"[S3ParquetIOManager] File does not exist: s3://{path} — returning None."
            )
            return None
