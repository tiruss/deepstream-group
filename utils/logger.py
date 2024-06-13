from __future__ import annotations

import logging
import multiprocessing
from logging.handlers import RotatingFileHandler
from typing import Any


class SafeRotatingFileHandler(RotatingFileHandler):
    """
    Multiprocess safe RotatingFileHandler
    taken from https://gist.github.com/SerhoLiu/a3d7be43df882af80ef98bc375fc6046
    """

    _rollover_lock = multiprocessing.Lock()

    def emit(self, record: Any) -> None:
        """Emit a record.

        Output the record to the file, catering for rollover as
        described in doRollover().
        """
        try:
            if self.shouldRollover(record):
                with self._rollover_lock:
                    if self.shouldRollover(record):
                        self.doRollover()
            logging.FileHandler.emit(self, record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)

    def shouldRollover(self, record: Any) -> Any:
        if self._should_rollover():
            # if some other process already did the rollover we might
            # checked log.1, so we reopen the stream and check again on
            # the right log file
            if self.stream:
                self.stream.close()
                self.stream = self._open()

            return self._should_rollover()

        return 0

    def _should_rollover(self) -> bool:
        if int(self.maxBytes) > 0:
            self.stream.seek(0, 2)
            if self.stream.tell() >= int(self.maxBytes):
                return True
        return False
