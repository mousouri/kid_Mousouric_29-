import logging
from typing import Optional
from config import Config

class Logger:
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('trading_bot')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(self.config.log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
        
    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
        
    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
        
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
        
    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)
        
    def critical(self, message: str) -> None:
        """Log critical message"""
        self.logger.critical(message) 