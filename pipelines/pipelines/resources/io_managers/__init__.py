from dagster import ConfigurableIOManager, InputContext, OutputContext
import pandas as pd
import s3fs

class S3CSVIOManager(ConfigurableIOManager):
    s3_bucket: str
    s3_prefix: str = "analytics"

    def _get_path(self, context: OutputContext | InputContext) -> str:
        asset_path = "/".join(context.asset_key.path)

        if context.has_asset_partitions:
            partition_key = context.partition_key or "unpartitioned"
            file_name = f"{partition_key}.csv"
        else:
            file_name = f"{asset_path}.csv"

        return f"{self.s3_bucket}/{self.s3_prefix}/{asset_path}/{file_name}"

    def handle_output(self, context: OutputContext, obj: pd.DataFrame):
        path = self._get_path(context)
        fs = s3fs.S3FileSystem()
        with fs.open(f"s3://{path}", "w") as f:
            obj.to_csv(f, index=False)

    def load_input(self, context: InputContext) -> pd.DataFrame:
        path = self._get_path(context.upstream_output)
        fs = s3fs.S3FileSystem()
        with fs.open(f"s3://{path}", "r") as f:
            return pd.read_csv(f)

class S3ParquetIOManager(ConfigurableIOManager):
    s3_bucket: str
    s3_prefix: str = "analytics"

    def _get_path(self, context: OutputContext | InputContext) -> str:
        asset_path = "/".join(context.asset_key.path)
            
        if context.has_asset_partitions:
            partition_key = context.partition_key or "unpartitioned"
            file_name = f"{partition_key}.parquet"
        else:
            file_name = f"{asset_path}.parquet"

        return f"{self.s3_bucket}/{self.s3_prefix}/{asset_path}/{file_name}"

    def handle_output(self, context: OutputContext, obj: pd.DataFrame):
        path = self._get_path(context)
        fs = s3fs.S3FileSystem()
        with fs.open(f"s3://{path}", "wb") as f:
            obj.to_parquet(f, index=False)

    def load_input(self, context: InputContext) -> pd.DataFrame:
        path = self._get_path(context.upstream_output)
        fs = s3fs.S3FileSystem()
        with fs.open(f"s3://{path}", "rb") as f:
            return pd.read_parquet(f)