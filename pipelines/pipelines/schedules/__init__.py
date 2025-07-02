from dagster import schedule, RunRequest, ScheduleEvaluationContext
from ..jobs import daily_trades_job, daily_scraping_job

@schedule(
    job=daily_trades_job, 
    cron_schedule="*/5 * * * *",
    execution_timezone="UTC"
)
def daily_schedule(context: ScheduleEvaluationContext):
    """Update our daily trades data every 5 minutes"""
    day_to_process = context.scheduled_execution_time.strftime("%Y-%m-%d")
    return RunRequest(run_key=None, partition_key=day_to_process)

@schedule(
    job=daily_scraping_job,
    cron_schedule="0 0 * * *", # Every day at midnight UTC
    execution_timezone="UTC"
)
def daily_scraping_schedule(_context: ScheduleEvaluationContext):
    """Scrape data from perpetualpulse.xyz daily."""
    return RunRequest(run_key=None)