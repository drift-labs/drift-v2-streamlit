from dagster import define_asset_job, AssetSelection
from ..partitions import daily_partitions, dlob_start_daily_partitions

daily_trades_job = define_asset_job(
    name="daily_trades_job",
    partitions_def=daily_partitions,
    selection=AssetSelection.keys("raw_trades") | AssetSelection.groups("trades"),
)


daily_trigger_order_report_job = define_asset_job(
    name="daily_trigger_order_report",
    partitions_def=dlob_start_daily_partitions,
    selection=(
        AssetSelection.keys("raw_trigger_orders", "raw_actions").downstream()
        | AssetSelection.keys("raw_trades", "clean_trades")
    ),
)
