from dagster import DailyPartitionsDefinition

daily_partitions = DailyPartitionsDefinition(start_date="2025-01-01")
dlob_start_daily_partitions = DailyPartitionsDefinition(start_date="2025-06-24")
