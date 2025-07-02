from dagster import Definitions, load_assets_from_modules, EnvVar
from .assets import raw_data, trades, trigger_orders, actions
from .resources import (
    AthenaConfig,
    WrappedAthenaClientResource,
    S3CSVIOManager,
    S3ParquetIOManager,
    LocalCSVIOManager
)
from .jobs import daily_trades_job, daily_trigger_order_report_job
from .schedules import daily_trades_schedule, daily_trigger_orders_schedule
import os

IO_MANAGERS = {
    "LOCAL": {
        "io_manager": LocalCSVIOManager(),
        "csv_io_manager": LocalCSVIOManager(),
    },
    "PROD": {
        "io_manager": S3ParquetIOManager(
            s3_bucket=EnvVar("S3_OUTPUT_BUCKET"),
            s3_prefix=EnvVar("S3_OUTPUT_PATH"),
        ),
        "csv_io_manager": S3CSVIOManager(
            s3_bucket=EnvVar("S3_OUTPUT_BUCKET"),
            s3_prefix=EnvVar("S3_OUTPUT_PATH"),
        ),
    },
}

environment = os.getenv("DAGSTER_ENV", "LOCAL")

defs = Definitions(
    assets=load_assets_from_modules([raw_data, trades, trigger_orders, actions]),
    jobs=[daily_trades_job, daily_trigger_order_report_job],
    schedules=[daily_trades_schedule, daily_trigger_orders_schedule],
    resources={
        **IO_MANAGERS[environment],
        "athena": WrappedAthenaClientResource(workgroup=EnvVar("ATHENA_WORKGROUP")),
        "athena_config": AthenaConfig(database_name=EnvVar("ATHENA_DB")),
    },
)
