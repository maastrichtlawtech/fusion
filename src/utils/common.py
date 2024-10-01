def load_sbert_model(model_name: str, max_seq_length: int, pooling: str, device: str = None):
    import os
    from dotenv import load_dotenv
    from sentence_transformers.models import Transformer, Pooling
    try:
        from src.utils.sentence_transformers import SentenceTransformerCustom
    except ModuleNotFoundError:
        import sys, pathlib
        sys.path.append(str(pathlib.Path().resolve()))
    from src.utils.sentence_transformers import SentenceTransformerCustom

    load_dotenv()
    embedding_model = Transformer(
        model_name_or_path=model_name,
        model_args={'token': os.getenv("HF")},
        max_seq_length=max_seq_length,
        tokenizer_args={'model_max_length': max_seq_length},
    )
    pooling_model = Pooling(word_embedding_dimension=embedding_model.get_word_embedding_dimension(), pooling_mode=pooling)
    return SentenceTransformerCustom(modules=[embedding_model, pooling_model], device=device)


class catchtime:
    from time import perf_counter
    
    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        self.readout = f'Time: {self.time:.3f} seconds'


def log_step(funct):
    """ 
    Decorator to log the time taken by a function to execute.
    """
    import timeit, datetime
    from functools import wraps

    @wraps(funct)
    def wrapper(*args, **kwargs):
        tic = timeit.default_timer()
        result = funct(*args, **kwargs)
        time_taken = datetime.timedelta(seconds=timeit.default_timer() - tic)
        print(f"- Just ran '{funct.__name__}' function. Took: {time_taken}")
        return result
    return wrapper


def read_json_file(path_or_url: str):
    """ 
    Read a JSON file from a local path or URL.
    """
    import re
    import json
    import urllib.request

    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    if bool(url_pattern.match(path_or_url)):
        with urllib.request.urlopen(path_or_url) as f:
            return json.load(f)
    with open(path_or_url, 'r') as f:
        return json.load(f)


def set_seed(seed: int):
    """ 
    Ensure that all operations are deterministic on CPU and GPU (if used) for reproducibility.
    """
    import torch
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def count_trainable_parameters(model, verbose=True):
    """ 
    Count the number of trainable parameters in a model.
    """
    all_params = 0
    trainable_params = 0
    for _, p in model.named_parameters():
        all_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    if verbose:
        print(f"Trainable params: {round(trainable_params/1e6, 1)}M || All params: {round(all_params/1e6, 1)}M || Trainable ratio: {round(100 * trainable_params / all_params, 2)}%")
    return trainable_params


def push_to_hub(ressource_type: str, ressource_path: str, username: str, repo_id: str, create_repo: bool, private_repo: bool = True):
    """ 
    """
    import os
    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    assert ressource_type in ["dataset", "model"], "ressource_type must be either 'dataset' or 'model'"
    try:
        load_dotenv()
        api = HfApi(token=os.getenv("HF"))
        if create_repo:
            api.create_repo(repo_id=repo_id, repo_type=ressource_type, private=private_repo)
        if os.path.isfile(ressource_path):
            api.upload_file(
                repo_id=f"{username}/{repo_id}",
                repo_type=ressource_type,
                path_or_fileobj=ressource_path,
                path_in_repo=os.path.basename(ressource_path),
            )
        elif os.path.isdir(ressource_path):
            api.upload_folder(
                repo_id=f"{username}/{repo_id}", 
                repo_type=ressource_type, 
                folder_path=ressource_path,
            )
    except Exception as e:
        print("An error occurred while uploading ressource to HuggingFace Hub:", str(e))


def download_if_not_exists(data_folder: str, file_url: str):
    """ 
    """
    from sentence_transformers import util
    from os.path import exists, join, basename

    save_path = join(data_folder, basename(file_url))
    if not exists(save_path):
        util.http_get(file_url, save_path)
    return save_path


def tsv_to_jsonl(tsv_filename: str):
    """ 
    """
    import json
    from tqdm import tqdm

    jsonl_filename = tsv_filename.replace('.tsv', '.jsonl')
    with open(tsv_filename, 'r') as fIn, open(jsonl_filename, 'w') as fOut:
        for line in tqdm(fIn, desc="Converting"):
            parts = [int(pid) for pid in line.strip().split('\t')]
            fOut.write(json.dumps(parts) + '\n')
    return jsonl_filename


def convert_colbert_results(results_filepath: str, output_filepath: str):
    """ 
    Convert the evaluation search results of ColBERT saved in tsv file where each line has the format "qid pid rank score" to
    a jsonl file where each line has the format {"qid": ..., "pos": [...], "neg": [...]}, following the msmarco-hard-negatives
    dataset (https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives).
    """
    import json
    import ir_datasets

    dev_samples = {}
    dev = ir_datasets.load("msmarco-passage/dev/small")
    for s in dev.qrels_iter():
        qid, pos_pid = int(s.query_id), int(s.doc_id)
        dev_samples.setdefault(qid, {'qid': qid, 'pos': [], 'neg': []})
        if pos_pid not in dev_samples[qid]['pos']:
            dev_samples[qid]['pos'].append(pos_pid)

    with open(results_filepath, 'r') as fIn:
        for line in fIn:
            qid, pid, rank, score = [int(x) if x.isdigit() else float(x) for x in line.strip().split()]
            if pid not in dev_samples[qid]['pos']:
                dev_samples[qid].setdefault('neg', []).append(pid)

    with open(output_filepath, 'w') as fOut:
        for qid, sample in dev_samples.items():
            fOut.write(json.dumps(sample) + '\n')


def get_training_filepath(data_dir: str, filename_pattern:str, target_num_examples: float):
    """ 
    """
    import glob
    from os.path import join, basename

    for filepath in glob.glob(join(data_dir, filename_pattern)):
        num = float(basename(filepath).split('.')[-2][:-1])
        if num >= target_num_examples:
            return filepath
    return join(data_dir, filename_pattern.replace('*', f'{target_num_examples:.1f}'))


def esimate_flops(model_name: str, seq_len: int, batch_size: int = 1):
    """ 
    Estimate the FLOPS of a model.
    """
    from transformers import AutoTokenizer
    from deepspeed.accelerator import get_accelerator
    from deepspeed.profiling.flops_profiler import get_model_profile

    with get_accelerator().device(0):
        
        model = load_sbert_model(model_name, max_seq_length=seq_len, pooling='mean')
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        fake_seq = ""
        for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
            fake_seq += tokenizer.pad_token
        inputs = tokenizer([fake_seq] * batch_size, padding=True, truncation=True, return_tensors="pt")

        flops, macs, params = get_model_profile(model, args=(inputs,), print_profile=True, detailed=True)
    
    return flops
