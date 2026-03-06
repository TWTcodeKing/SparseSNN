"""
Pretty logger for training and evaluation progress.
Uses dashed-line formatting for clear visual separation.
"""

import sys
import time
import logging
import datetime


def setup_logger(name, log_file=None, level=logging.INFO):
    """Create a logger with console + optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, mode='a')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


_DASH = "-" * 72


class TrainLogger:
    """
    Structured logger for training loops.

    Usage:
        tlog = TrainLogger(logger, total_epochs=100)
        tlog.epoch_start(epoch)
        for step, (loss, acc) in enumerate(train_loop):
            tlog.step(epoch, step, total_steps, loss=loss, acc1=acc)
        tlog.epoch_end(epoch, train_metrics, val_metrics)
    """

    def __init__(self, logger, total_epochs, print_freq=50):
        self.logger = logger
        self.total_epochs = total_epochs
        self.print_freq = print_freq
        self._epoch_t0 = None

    def banner(self, msg):
        self.logger.info(_DASH)
        self.logger.info(msg)
        self.logger.info(_DASH)

    def epoch_start(self, epoch):
        self._epoch_t0 = time.time()
        self.logger.info(_DASH)
        self.logger.info(
            f"Epoch [{epoch + 1}/{self.total_epochs}]  START"
        )
        self.logger.info(_DASH)

    def step(self, epoch, step, total_steps, **metrics):
        if (step + 1) % self.print_freq != 0 and step != total_steps - 1:
            return
        parts = [f"Epoch [{epoch + 1}/{self.total_epochs}]"
                 f"  Step [{step + 1}/{total_steps}]"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}: {v:.4f}")
            else:
                parts.append(f"{k}: {v}")
        self.logger.info("  ".join(parts))

    def epoch_end(self, epoch, train_metrics=None, val_metrics=None):
        elapsed = time.time() - self._epoch_t0 if self._epoch_t0 else 0
        self.logger.info(_DASH)
        line = f"Epoch [{epoch + 1}/{self.total_epochs}]  END  " \
               f"({str(datetime.timedelta(seconds=int(elapsed)))})"
        self.logger.info(line)
        if train_metrics:
            parts = ["  Train >>"]
            for k, v in train_metrics.items():
                parts.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
            self.logger.info("  ".join(parts))
        if val_metrics:
            parts = ["  Val   >>"]
            for k, v in val_metrics.items():
                parts.append(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
            self.logger.info("  ".join(parts))
        self.logger.info(_DASH)

    def best(self, epoch, acc):
        self.logger.info(
            f"  *** New Best Acc@1: {acc:.4f} at Epoch {epoch + 1} ***"
        )

    def finish(self, best_acc, total_time):
        self.logger.info("")
        self.logger.info("=" * 72)
        self.logger.info(
            f"Training Finished.  Best Acc@1: {best_acc:.4f}  "
            f"Total Time: {str(datetime.timedelta(seconds=int(total_time)))}"
        )
        self.logger.info("=" * 72)
