from dagster import define_asset_job, AssetSelection
from ..partitions import daily_partitions

daily_trades_job = define_asset_job(
    name="daily_trades_job",
    partitions_def=daily_partitions,
    selection=AssetSelection.keys('raw_trades').downstream()
)
