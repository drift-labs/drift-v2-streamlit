from dagster import Definitions, load_assets_from_modules, EnvVar
from .assets import raw_data, trades
from .resources import AthenaConfig, WrappedAthenaClientResource, S3CSVIOManager, S3ParquetIOManager
from .jobs import daily_trades_job
from .schedules import daily_schedule

defs = Definitions(
    assets=load_assets_from_modules([raw_data, trades]),
    jobs=[daily_trades_job],
    schedules=[daily_schedule],
    resources={
        "io_manager": S3ParquetIOManager(
            s3_bucket=EnvVar("S3_OUTPUT_BUCKET"),
            s3_prefix=EnvVar("S3_OUTPUT_PATH"),
        ),
        "csv_io_manager": S3CSVIOManager(
            s3_bucket=EnvVar("S3_OUTPUT_BUCKET"),
            s3_prefix=EnvVar("S3_OUTPUT_PATH"),
        ),
        "athena": WrappedAthenaClientResource(
            workgroup=EnvVar("ATHENA_WORKGROUP")
        ),
        "athena_config": AthenaConfig(
            database_name=EnvVar("ATHENA_DB")
        )
    }
)
