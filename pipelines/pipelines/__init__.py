from dagster import Definitions, load_assets_from_modules, EnvVar
from .assets import raw_data, trades, reports
from .resources import AthenaConfig, WrappedAthenaClientResource, S3CSVIOManager, S3ParquetIOManager
from .jobs import daily_trades_job, daily_scraping_job
from .schedules import daily_schedule, daily_scraping_schedule

defs = Definitions(
    assets=load_assets_from_modules([raw_data, trades, reports]),
    jobs=[daily_trades_job, daily_scraping_job],
    schedules=[daily_schedule, daily_scraping_schedule],
    resources={
        "io_manager": S3ParquetIOManager(
            s3_bucket=EnvVar("S3_OUTPUT_BUCKET"),
            s3_prefix=EnvVar("S3_OUTPUT_PATH"),
        ),
        "csv_io_manager": S3CSVIOManager(
            s3_bucket=EnvVar("S3_OUTPUT_BUCKET"),
            s3_prefix=EnvVar("S3_OUTPUT_PATH"),
        ),
        "athena": WrappedAthenaClientResource(),
        "athena_config": AthenaConfig(
            database_name=EnvVar("ATHENA_DATABASE_NAME"),
        )
    }
)
