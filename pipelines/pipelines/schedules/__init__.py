from dagster import schedule, RunRequest, ScheduleEvaluationContext, build_schedule_from_partitioned_job
from ..jobs import daily_trades_job, daily_trigger_order_report_job

# This automatically creates a schedule that runs for each partition
daily_trigger_orders_schedule = build_schedule_from_partitioned_job(
    daily_trigger_order_report_job,
)

@schedule(job=daily_trades_job, cron_schedule="*/5 * * * *", execution_timezone="UTC")
def daily_schedule(context: ScheduleEvaluationContext):
    """Update our daily trades data every 5 minutes"""
    day_to_process = context.scheduled_execution_time.strftime("%Y-%m-%d")
    return RunRequest(run_key=None, partition_key=day_to_process)
