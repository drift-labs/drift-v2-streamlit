from dagster import schedule, RunRequest, ScheduleEvaluationContext, build_schedule_from_partitioned_job
from ..jobs import daily_trades_job, daily_trigger_order_report_job

daily_trigger_orders_schedule = build_schedule_from_partitioned_job(
    daily_trigger_order_report_job,
)

daily_trades_schedule = build_schedule_from_partitioned_job(
    daily_trades_job,
)