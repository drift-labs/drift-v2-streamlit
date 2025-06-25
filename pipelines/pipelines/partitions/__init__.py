from dagster import DailyPartitionsDefinition

daily_partitions = DailyPartitionsDefinition(start_date="2025-06-10", end_offset=1)
