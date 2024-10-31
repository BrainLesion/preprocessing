import sys
import signal
import logging
import traceback

from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class LoggingManager:
    """
    Manages logging configurations for the application or library.
    """

    def __init__(self, name: str, log_file_path: Optional[Union[str, Path]] = None):
        self.name = name
        self.log_file_path = log_file_path

        # Create and configure the custom logger
        self.logger = logging.getLogger(self.name)
        self.log_file_handler = None

        # Disable log propagation
        self.logger.propagate = False

        # Set up the logger
        self._setup_logger()

        # Set up the logger file if provided
        if log_file_path:
            self._set_log_file(log_file_path)

    def get_logger(self) -> logging.Logger:
        """
        Returns the logger instance.
        """
        return self.logger

    def _setup_logger(self):
        """
        Sets up the custom logger and overwrites system hooks to add logging for exceptions and signals.
        """
        # Configure the custom logger if it hasn't been configured yet
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "[%(levelname)s | %(name)s] %(asctime)s: %(message)s",
                "%Y-%m-%dT%H:%M:%S%z",
            )

            # Add a console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Overwrite system hooks to log exceptions and signals (caution advised)
        sys.excepthook = self.exception_handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def _set_log_file(self, log_file: str = None) -> None:
        """
        Sets the log file handler for the logger.

        Args:
            log_file (str | Path, optional): Log file path. If not provided, a timestamped log file is created.
        """

        # Remove existing log file handler if present
        if self.log_file_handler:
            self.remove_log_file_handler()

        # Ensure parent directories exist
        log_file = Path(
            log_file
            if log_file
            else f"brainles_preprocessing_{datetime.now().strftime('%Y-%m-%d_T%H-%M-%S.%f')}.log"
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create and add the file handler
        self.log_file_handler = logging.FileHandler(str(log_file))
        self.log_file_handler.setFormatter(
            logging.Formatter(
                "[%(levelname)-8s | %(name)s | %(module)-15s | L%(lineno)-5d] %(asctime)s: %(message)s",
                "%Y-%m-%dT%H:%M:%S%z",
            )
        )
        self.logger.addHandler(self.log_file_handler)
        self.log_file_path = log_file

    def remove_log_file_handler(self) -> None:
        """
        Removes the log file handler from the logger.
        """
        if self.log_file_handler:
            self.logger.removeHandler(self.log_file_handler)
            self.log_file_handler.close()
            self.log_file_handler = None
            self.log_file_path = None

    # overwrite system hooks to log exceptions and signals (SIGINT, SIGTERM)
    #! NOTE: This will note work in Jupyter Notebooks, (Without extra setup) see https://stackoverflow.com/a/70469055:
    def exception_handler(self, exception_type, value, tb):
        """Handle exceptions

        Args:
            exception_type (Exception): Exception type
            exception (Exception): Exception
            traceback (Traceback): Traceback
        """
        self.logger.error(
            "".join(traceback.format_exception(exception_type, value, tb))
        )

        if issubclass(exception_type, SystemExit):
            # add specific code if exception was a system exit
            sys.exit(value.code)

    def signal_handler(self, sig, frame):
        """
        Handles signals by logging them and exiting.

        Args:
            sig (int): Signal number
            frame (FrameType): Current stack frame
        """
        signame = signal.Signals(sig).name
        self.logger.error(f"Received signal {sig} ({signame}), exiting...")
        sys.exit(0)
