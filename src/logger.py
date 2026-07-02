import logging
import os

def setup_logger(name="ml_pipeline", log_file="logs/pipeline.log", level=logging.INFO):
    """Configures and returns a logger that outputs to both console and a log file."""
    # Ensure logs directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger is already initialized
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] (%(name)s) %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

# Create a default logger instance
logger = setup_logger()
