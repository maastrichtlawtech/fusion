import wandb
import logging
from tqdm import tqdm


class WandbLogger:
    """
    Wandb logger for logging training and evaluation metrics.
    Inspired from: https://github.com/UKPLab/sentence-transformers/issues/705#issuecomment-833521213

    :param project_name: The name of the project.
    :param run_name: The name of the run.
    :param run_config: The configuration for the run.
    :param log_dir: The directory to store the logs.
    """
    def __init__(self, project_name: str, run_name: str, run_config: dict, log_dir: str):
        if wandb.run is not None:
            self.experiment = wandb.run
        else:
            self.experiment = wandb.init(project=project_name, name=run_name, dir=log_dir, config=run_config)

    def log_training(self, train_idx: int, epoch: int, global_step: int, current_lr: float, loss_value: float, loss_name: str = "loss"):
        """
        Log training metrics.

        :param train_idx: The index of the training step.
        :param epoch: The current epoch.
        :param global_step: The global step.
        :param current_lr: The current learning rate.
        :param loss_value: The loss value.
        """
        self.experiment.log(step=global_step, data={
            f"train/{loss_name}": loss_value, 
            "train/lr": current_lr, 
            "train/epoch": epoch,
        })

    def log_eval(self, epoch: int, global_step: int, prefix: str, value: float):
        """
        Log evaluation metrics.

        :param epoch: The current epoch.
        :param global_step: The global step.
        :param prefix: The prefix for the metric.
        :param value: The metric value.
        """
        self.experiment.log(step=global_step, data={prefix: value})

    def finish(self):
        """Finish the logging session."""
        self.experiment.finish()


class LoggingHandler(logging.Handler):
    """
    Custom logging handler for logging to tqdm.
    Inspired from: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/LoggingHandler.py

    :param level: The logging level.
    """
    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)

    def emit(self, record: logging.LogRecord):
        """
        Emit the log record.

        :param record: The log record.
        """
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
