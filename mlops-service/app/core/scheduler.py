from apscheduler.schedulers.background import BackgroundScheduler
from .service import train_model, check_drift
from .config import settings
import logging

logger = logging.getLogger(__name__)
_scheduler = None

def _auto_retrain():
    drift = check_drift()
    if drift["drift_detected"]:
        logger.info(f"Drift detected ({drift['drift_score']:.3f}), triggering retraining...")
        result = train_model()
        logger.info(f"Auto-retrain complete: accuracy={result['accuracy']:.3f}")

def start_scheduler():
    global _scheduler
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(_auto_retrain, "interval", hours=settings.RETRAIN_SCHEDULE_HOURS, id="auto_retrain")
    _scheduler.start()
    logger.info(f"Scheduler started — retraining every {settings.RETRAIN_SCHEDULE_HOURS}h")

def stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown()
