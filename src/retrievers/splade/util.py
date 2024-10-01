import os
import sys
import tqdm
import random
import warnings
import requests
import torch
import numpy as np

__all__ = ["batchify", "download_if_not_exists", "duplicates_queries_warning", "set_seed"]


def set_seed(seed: int):
    """ 
    Ensure that all operations are deterministic on CPU and GPU (if used) for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def duplicates_queries_warning() -> None:
    message = """Duplicate queries found. Provide distinct queries."""
    warnings.warn(message=message)


def batchify(X: list[str], batch_size: int, tqdm_bar: bool = True, tqdm_msg: str = "") -> list:
    """ 
    Batchify a list of strings.

    :param X: List of strings.
    :param batch_size: Batch size.
    :param desc: Description for the tqdm bar.
    :param tqdm_bar: Whether to display a tqdm bar or not.
    :returns: Returns a list of batches.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    batchs = [X[pos:pos+batch_size] for pos in range(0, len(X), batch_size)]
    if tqdm_bar:
        for batch in tqdm.tqdm(batchs, position=0, total=len(X)//batch_size, desc=tqdm_msg):
            yield batch
    else:
        yield from batchs


def download_if_not_exists(data_dir: str, file_url: str) -> str:
    """
    Downloads a URL to a given path on disc.

    :param data_dir: Directory to save the file.
    :param file_url: URL to download.
    :returns: Path to the downloaded file.
    """
    path = os.path.join(data_dir, os.path.basename(file_url))
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)

        req = requests.get(file_url, stream=True)
        if req.status_code != 200:
            print("Exception when trying to download {}. Response {}".format(file_url, req.status_code), file=sys.stderr)
            req.raise_for_status()
            return

        download_filepath = path + "_part"
        with open(download_filepath, "wb") as file_binary:
            content_length = req.headers.get("Content-Length")
            total = int(content_length) if content_length is not None else None
            progress = tqdm.tqdm(unit="B", total=total, unit_scale=True)
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    file_binary.write(chunk)

        os.rename(download_filepath, path)
        progress.close()
    return path
