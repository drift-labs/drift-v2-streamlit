from dagster import schedule, RunRequest, ScheduleEvaluationContext
from ..jobs import daily_trades_job

@schedule(
    job=daily_trades_job, 
    cron_schedule="* * * * *",
    execution_timezone="UTC"
)
def daily_schedule(context: ScheduleEvaluationContext):
    """Update our daily trades data every minute"""
    day_to_process = context.scheduled_execution_time.strftime("%Y-%m-%d")
    return RunRequest(run_key=None, partition_key=day_to_process)