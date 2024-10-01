"""
The MS MARCO dataset is a large-scale IR dataset from Microsoft Bing comprising:
- a corpus of 8.8M passages;
- a training set of ~533k queries (with at least one relevant passage);
- a development set of ~101k queries;
- a smaller dev set of 6,980 queries (which is actually used for evaluation in most published works).
The mMARCO dataset is a machine-translated version of MS Marco in 13 different languages.
Link: https://ir-datasets.com/mmarco.html#mmarco/v2/fr/.
"""
import tqdm
import random
import json, gzip, pickle
from os.path import exists
from typing import Dict, Optional, Type

import ir_datasets
from torch.utils.data import Dataset
from sentence_transformers import InputExample

try:
    from src.utils.common import MMARCO_LANGUAGES, download_if_not_exists, tsv_to_jsonl, get_training_filepath
except ModuleNotFoundError:
    import sys, pathlib
    sys.path.append(str(pathlib.Path().resolve()))
from src.utils.common import MMARCO_LANGUAGES, download_if_not_exists, tsv_to_jsonl, get_training_filepath


NEGS_MINING_SYSTEMS = [
    # https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives
    'bm25', 
    'msmarco-distilbert-base-tas-b',
    'msmarco-distilbert-base-v3',
    'msmarco-MiniLM-L-6-v3',
    'distilbert-margin_mse-cls-dot-v2',
    'distilbert-margin_mse-cls-dot-v1',
    'distilbert-margin_mse-mean-dot-v1',
    'mpnet-margin_mse-mean-v1',
    'co-condenser-margin_mse-cls-v1',
    'distilbert-margin_mse-mnrl-mean-v1',
    'distilbert-margin_mse-sym_mnrl-mean-v1',
    'distilbert-margin_mse-sym_mnrl-mean-v2',
    'co-condenser-margin_mse-sym_mnrl-mean-v1',
]

class MmarcoColbertLoader:
    """
    Data loader for sampling the MS MARCO dataset to train/evaluate ColBERT models.

    :param lang: Language in which MS Marco is loaded.
    :param load_dev: Whether to load the small dev set of MS MARCO.
    :param load_train: Whether to load the training data.
    :param max_train_examples: Maximum number of training samples to sample. The official training set contains 808731 queries, yet only 502939 have associated labels.
    :param training_type: Type of training samples to use ("v1" or "v2"). The former are official MS MARCO training triples with a single BM25 negative. 
        The latter are custom training samples obtained by mining hard negatives from dense retrievers.
    :param negs_mining_systems: Comma-separated list of systems used for mining hard negatives. Only required if training_type == 'v2'.
    :param negs_per_query: Number of hard negatives to sample per query. Only required if training_type == 'v2'.
    :param ce_score_margin: Margin for the cross-encoder score between negative and positive passages. Only required if training_type == 'v2'.
    :param data_folder: Folder in which to save the downloaded datasets.
    """
    def __init__(
        self, 
        lang: str,  # 
        load_dev: Optional[str] = True,
        load_train: Optional[str] = True,
        max_train_examples: Optional[int] = 502939,
        training_type: Optional[str] = 'v2',
        negs_mining_systems: Optional[str] = '',
        negs_per_query: Optional[int] = 63,
        ce_score_margin: Optional[float] = 3.0,
        data_folder: Optional[str] = 'data/mmarco',
    ):
        assert lang in MMARCO_LANGUAGES.keys(), f"Language {lang} not supported."
        assert training_type in ["v1", "v2"], f"Unkwown type of training qrels. Please choose between 'v1' and 'v2'."
        if negs_mining_systems:
            assert all(syst in NEGS_MINING_SYSTEMS for syst in negs_mining_systems.split(',')), f"Unknown hard negative mining system."
        self.lang = lang
        self.load_dev = load_dev
        self.load_train = load_train
        self.max_train_examples = max_train_examples
        self.training_type = training_type
        self.negs_mining_systems = negs_mining_systems
        self.negs_per_query = negs_per_query
        self.ce_score_margin = ce_score_margin
        self.data_folder = data_folder
    
    def load(self) -> Dict[str, str]:
        """
        Returns a dictionary containing the filepaths to the datasets. 
        For a v1 training type, the training tuples will be of the form (qid, pos_pid, neg_pid).
        For a v2 training type, the training tuples will be of the form [qid, (pos_pid, pos_score), (neg_pid1, neg_score1), ..., (neg_pidN, neg_scoreN)].
        """
        data_filepaths = {}

        # Load collection of passages.
        url = f'https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/{MMARCO_LANGUAGES.get(self.lang)[0]}_collection.tsv'
        data_filepaths['collection'] = download_if_not_exists(data_folder=self.data_folder, file_url=url)

        if self.load_dev:
            # Load dev queries. 
            url = f'https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/{MMARCO_LANGUAGES.get(self.lang)[0]}_queries.dev.small.tsv'
            data_filepaths['dev_queries'] = download_if_not_exists(data_folder=self.data_folder, file_url=url)

            # Load dev qrels. 
            url = 'https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/qrels.dev.small.tsv'
            data_filepaths['dev_qrels'] = download_if_not_exists(data_folder=self.data_folder, file_url=url)

        if self.load_train:
            # Load training queries. 
            url = f'https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/{MMARCO_LANGUAGES.get(self.lang)[0]}_queries.train.tsv'
            data_filepaths['train_queries'] = download_if_not_exists(data_folder=self.data_folder, file_url=url)

            # Load training qrels. 
            if self.training_type == "v1":
                url = 'https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/triples.train.ids.small.tsv'
                save_path = download_if_not_exists(data_folder=self.data_folder, file_url=url)
                data_filepaths['train_tuples'] = tsv_to_jsonl(save_path)
            else:
                training_tuples_filepath = get_training_filepath(
                    data_dir=self.data_folder, 
                    filename_pattern=f'tuples.train.qid-pos-negs.{self.negs_per_query+1}way.*M.jsonl', 
                    target_num_examples=round(self.max_train_examples/1e6, 1),
                )
                if not exists(training_tuples_filepath):
                    # Load CE scores for query-passage pairs: ce_scores[qid][pid] -> score.
                    url = 'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz'
                    ce_scores_file = download_if_not_exists(data_folder=self.data_folder, file_url=url)
                    with gzip.open(ce_scores_file, 'rb') as fIn:
                        ce_scores = pickle.load(fIn)

                    # Load hard negatives mined from BM25 and 12 different dense retrievers.
                    url = 'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz'
                    hard_negatives_filepath = download_if_not_exists(data_folder=self.data_folder, file_url=url)
                    num_examples, i = 0, 0
                    with gzip.open(hard_negatives_filepath, 'rt') as fIn, open(training_tuples_filepath, 'w') as fOut:
                        while num_examples < self.max_train_examples:
                            i += 1
                            fIn.seek(0)
                            random.seed(num_examples)
                            for line in tqdm.tqdm(fIn, desc=f"Sampling (qid, pos, negs) training tuples -> Dataset pass {i}"):
                                # Load the training sample: {"qid": ..., "pos": [...], "neg": {"system1": [...], "system2": [...], ...}}
                                data = json.loads(line)
                                qid, pos_pids = data['qid'], data['pos']
                                if len(pos_pids) == 0:
                                    continue

                                # Set the CE threshold as the minimum positive score minus a margin.
                                pos_min_ce_score = min([ce_scores[qid][pid] for pid in pos_pids])
                                ce_score_threshold = pos_min_ce_score - self.ce_score_margin

                                # Sample one positive passage and its associated CE scores.
                                sampled_pos_pid = random.choice(pos_pids)
                                sampled_pos_score = ce_scores[qid][sampled_pos_pid]
                                sampled_pos_tuple = [[sampled_pos_pid, sampled_pos_score]]

                                # Sample N hard negatives and their CE scores.
                                neg_pids = []
                                neg_systems = self.negs_mining_systems.split(",") if self.negs_mining_systems else list(data['neg'].keys())
                                for system_name in neg_systems:
                                    neg_pids.extend(data['neg'][system_name])
                                filtered_neg_pids = [pid for pid in list(set(neg_pids)) if ce_scores[qid][pid] <= ce_score_threshold]
                                sampled_neg_pids = random.sample(filtered_neg_pids, min(self.negs_per_query, len(filtered_neg_pids)))
                                sampled_neg_scores = [ce_scores[qid][pid] for pid in sampled_neg_pids]
                                sampled_neg_tuples = [list(pair) for pair in zip(sampled_neg_pids, sampled_neg_scores)]

                                if len(sampled_neg_tuples) == self.negs_per_query:
                                    sample = [qid] + sampled_pos_tuple + sampled_neg_tuples
                                    fOut.write(json.dumps(sample) + '\n')
                                    num_examples += 1
                                
                                if num_examples >= self.max_train_examples:
                                    break

                        del ce_scores
                        print(f"#> Number of training examples created: {num_examples}.")

                data_filepaths['train_tuples'] = training_tuples_filepath
                    
        return data_filepaths


class MmarcoCrossencoderLoader:
    """
    Data loader for sampling the MS MARCO dataset to train/evaluate cross-encoder models.
    Inspired from: https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_cross-encoder_scratch.py

    :param lang: Language in which MS Marco is loaded.
    :param load_dev: Whether to load the small dev set of MS MARCO.
    :param load_train: Whether to load the training data.
    :param max_train_examples: Maximum number of training samples to sample. The official training set contains 808731 queries, yet only 502939 have associated labels.
    :param negs_mining_systems: Comma-separated list of systems used for mining hard negatives.
    :param pos_neg_ratio: Positive-to-negative ratio in our training setup for the binary label task. For 1 positive sample (label 1), we sample X negative samples (label 0).
    :param ce_score_margin: Margin for the cross-encoder score between negative and positive passages.
    :param data_folder: Folder in which to save the downloaded datasets.
    """
    def __init__(
        self, 
        lang: str,
        load_dev: Optional[str] = False,
        load_train: Optional[str] = False,
        max_train_examples: Optional[int] = 502939,
        negs_mining_systems: Optional[str] = '',
        pos_neg_ratio: int = 1,
        ce_score_margin: Optional[float] = 3.0,
        data_folder: Optional[str] = 'data/mmarco',  # Folder in which to save the downloaded datasets.
    ):
        assert lang in MMARCO_LANGUAGES.keys(), f"Language {lang} not supported."
        if negs_mining_systems:
            assert all(syst in NEGS_MINING_SYSTEMS for syst in negs_mining_systems.split(',')), f"Unknown hard negative mining system."
        self.lang = lang.replace('nl', 'dt')
        self.load_dev = load_dev
        self.load_train = load_train
        self.max_train_examples = max_train_examples
        self.negs_mining_systems = negs_mining_systems
        self.pos_neg_ratio = pos_neg_ratio
        self.ce_score_margin = ce_score_margin
        self.data_folder = data_folder

    def load(self) -> Dict[str, object]:
        # Load collection of passages.
        collection = "msmarco-passage" if self.lang == "en" else f"mmarco/v2/{self.lang}"
        corpus = ir_datasets.load(collection)
        passages = {int(d.doc_id): d.text for d in corpus.docs_iter()}

        if self.load_dev:
            # Load dev queries.
            dev = ir_datasets.load(f"{collection}/dev/small")
            dev_queries = {int(q.query_id): q.text for q in dev.queries_iter()}

            # Create dev samples.
            url = 'https://huggingface.co/datasets/antoinelouis/msmarco-dev-small-negatives/resolve/main/colbertv2-negatives.dev.small.jsonl'
            negatives_filepath = download_if_not_exists(data_folder=self.data_folder, file_url=url)
            dev_samples = {}
            with open(negatives_filepath, 'r') as fIn:
                for line in tqdm.tqdm(fIn, desc="Loading dev samples"):
                    data = json.loads(line) #{"qid": ..., "pos": [...], "neg": [...]}
                    dev_samples[data['qid']] = {
                        "query": dev_queries[data['qid']],
                        "positive": {passages[pid] for pid in data['pos']},
                        "negative": {passages[pid] for pid in data['neg']},
                    }

        if self.load_train:
            # Load training queries.
            train = ir_datasets.load(f"{collection}/train")
            train_queries = {int(q.query_id): q.text for q in train.queries_iter()}

            # Create training samples.
            training_pairs_filepath = get_training_filepath(
                data_dir=self.data_folder, 
                filename_pattern='pairs.train.qid-pid-rel.*M.jsonl', 
                target_num_examples=round(self.max_train_examples/1e6, 1),
            )
            if exists(training_pairs_filepath):
                num_examples = 0
                train_samples = []
                with open(training_pairs_filepath, 'r') as fIn:
                    for line in tqdm.tqdm(fIn, desc="Loading (qid, pid, rel) training pairs"):
                        qid, pid, rel = json.loads(line)
                        train_samples.append(InputExample(texts=[train_queries[qid], passages[pid]], label=rel))
                        num_examples += 1
                        if num_examples >= self.max_train_examples:
                            break
            else:
                # Load CE scores for query-passage pairs: ce_scores[qid][pid] -> score.
                url = 'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz'
                ce_scores_file = download_if_not_exists(data_folder=self.data_folder, file_url=url)
                with gzip.open(ce_scores_file, 'rb') as fIn:
                    ce_scores = pickle.load(fIn)

                # Use hard negatives mined from BM25 and 12 different dense retrievers.
                url = 'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz'
                hard_negatives_filepath = download_if_not_exists(data_folder=self.data_folder, file_url=url)
                i = 0
                num_examples = 0
                train_samples = []
                with gzip.open(hard_negatives_filepath, 'rt') as fIn, open(training_pairs_filepath, 'w') as fOut:
                    while num_examples < self.max_train_examples:
                        i += 1
                        fIn.seek(0)
                        random.seed(num_examples)
                        for line in tqdm.tqdm(fIn, desc=f"Sampling (qid, pid, rel) training tuples -> Dataset pass {i}"):
                            # Load the training sample: {"qid": ..., "pos": [...], "neg": {"system1": [...], "system2": [...], ...}}
                            data = json.loads(line)
                            qid, pos_pids = data['qid'], data['pos']
                            if len(pos_pids) == 0:
                                continue
                            query = train_queries[qid]

                            # Set the CE threshold as the minimum positive score minus a margin.
                            pos_min_ce_score = min([ce_scores[qid][pid] for pid in pos_pids])
                            ce_score_threshold = pos_min_ce_score - self.ce_score_margin

                            if (num_examples % (self.pos_neg_ratio + 1)) == 0:
                                # Sample a positive passage.
                                sampled_pid = random.choice(pos_pids)
                                passage, label = passages[sampled_pid], 1
                            else:
                                # Sample a hard negative passage.
                                neg_pids = []
                                neg_systems = self.negs_mining_systems.split(",") if self.negs_mining_systems else list(data['neg'].keys())
                                for system_name in neg_systems:
                                    neg_pids.extend(data['neg'][system_name])
                                filtered_neg_pids = [pid for pid in list(set(neg_pids)) if ce_scores[qid][pid] <= ce_score_threshold]
                                try:
                                    sampled_pid = random.choice(filtered_neg_pids)
                                except IndexError:
                                    continue #filtered list is empty
                                passage, label = passages[sampled_pid], 0

                            train_samples.append(InputExample(texts=[query, passage], label=label))
                            fOut.write(json.dumps([qid, sampled_pid, label]) + '\n')
                            num_examples += 1
                            if num_examples >= self.max_train_examples:
                                break

                    del ce_scores

        return {
            'train': train_samples if self.load_train else None,
            'dev': dev_samples if self.load_dev else None,
            'test': None,
            'corpus': passages,
        }


class MmarcoBiencoderLoader:
    """
    Data loader for sampling the MS MARCO dataset to train/evaluate bi-encoder models.
    Inspired from: https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder_mnrl.py

    :param lang: Language in which MS Marco is loaded.
    :param load_dev: Whether to load the small dev set of MS MARCO.
    :param load_train: Whether to load the training data.
    :param negs_mining_systems: Comma-separated list of systems used for mining hard negatives.
    :param num_negs_per_system: Number of negatives to use per system.
    :param ce_score_margin: Margin for the cross-encoder score between negative and positive passages.
    :param data_folder: Folder in which to save the downloaded datasets.
    """
    def __init__(
        self, 
        lang: str,
        load_dev: Optional[str] = False,
        load_train: Optional[str] = False,
        negs_mining_systems: Optional[str] = '',
        num_negs_per_system: Optional[int] = 10,
        ce_score_margin: Optional[float] = 3.0,
        data_folder: Optional[str] = 'data/mmarco',
    ):
        assert lang in MMARCO_LANGUAGES.keys(), f"Language {lang} not supported."
        if negs_mining_systems:
            assert all(syst in NEGS_MINING_SYSTEMS for syst in negs_mining_systems.split(',')), f"Unknown hard negative mining system."
        self.lang = lang.replace('nl', 'dt')
        self.load_train = load_train
        self.load_dev = load_dev
        self.negs_mining_systems = negs_mining_systems
        self.num_negs_per_system = num_negs_per_system
        self.ce_score_margin = ce_score_margin
        self.data_folder = data_folder

    def load(self) -> Dict[str, object]:
        # Load collection of passages.
        collection = "msmarco-passage" if self.lang == "en" else f"mmarco/v2/{self.lang}"
        corpus = ir_datasets.load(collection)
        passages = {int(d.doc_id): d.text for d in corpus.docs_iter()}

        if self.load_dev:
            # Load dev queries.
            dev = ir_datasets.load(f"{collection}/dev/small")
            dev_queries = {int(q.query_id): q.text for q in dev.queries_iter()}

            # Load dev qrels.
            dev_qrels = {}
            for sample in dev.qrels_iter():
                dev_qrels.setdefault(int(sample.query_id), []).append(int(sample.doc_id))

        if self.load_train:
            random.seed(42)

            # Load training queries.
            train = ir_datasets.load(f"{collection}/train")
            train_queries = {int(q.query_id): q.text for q in train.queries_iter()}

            # Load CE scores for query-passage pairs: ce_scores[qid][pid] -> score.
            url = 'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz'
            ce_scores_file = download_if_not_exists(data_folder=self.data_folder, file_url=url)
            with gzip.open(ce_scores_file, 'rb') as fIn:
                ce_scores = pickle.load(fIn)

            # Load hard negatives mined from BM25 and 12 different dense retrievers.
            url = 'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz'
            hard_negatives_filepath = download_if_not_exists(data_folder=self.data_folder, file_url=url)
            train_samples = {}
            with gzip.open(hard_negatives_filepath, 'rt') as fIn:
                for line in tqdm.tqdm(fIn, desc=f"Sampling training tuples"):
                    # Load the training sample: {"qid": ..., "pos": [...], "neg": {"system1": [...], "system2": [...], ...}}
                    data = json.loads(line)
                    qid, pos_pids = data['qid'], data['pos']
                    if len(pos_pids) == 0:
                        continue
                    
                    # Set the CE threshold as the minimum positive score minus a margin.
                    pos_min_ce_score = min([ce_scores[qid][pid] for pid in pos_pids])
                    ce_score_threshold = pos_min_ce_score - self.ce_score_margin

                    # Sample k hard negatives per system (13 systems in total).
                    neg_pids = set()
                    neg_systems = self.negs_mining_systems.split(",") if self.negs_mining_systems else list(data['neg'].keys())
                    for system_name in neg_systems:
                        negs_added = 0
                        for pid in data['neg'][system_name]:
                            if ce_scores[qid][pid] > ce_score_threshold:
                                continue
                            if pid not in neg_pids:
                                neg_pids.add(pid)
                                negs_added += 1
                                if negs_added >= self.num_negs_per_system:
                                    break
                    
                    if len(pos_pids) > 0 and len(neg_pids) > 0:
                        train_samples[data['qid']] = {'qid': qid, 'query': train_queries[qid], 'pos': pos_pids, 'neg': neg_pids}
        
            del ce_scores
            train_dataset = MmarcoDataset(samples=train_samples, corpus=passages)

        return {
            'train': train_dataset if self.load_train else None,
            'dev': {'queries': dev_queries, 'labels': dev_qrels} if self.load_dev else None,
            'test': None,
            'corpus': passages, 
        }


class MmarcoDataset(Dataset):
    """
    Pytorch Dataset for training bi-encoder models on the MS MARCO dataset.
    Inspired from: https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder_mnrl.py

    :param samples: Dictionary containing the training samples in the form {qid: {"qid": ..., "query": ..., "pos": [...], "neg": [...]}}.
    :param corpus: Dictionary containing the collection of passages in the form {pid: passage}.
    """
    def __init__(self, samples: dict, corpus: dict):
        self.samples = samples
        self.corpus = corpus
        self.sample_ids = list(samples.keys())
        for qid in self.samples:
            self.samples[qid]['pos'] = list(self.samples[qid]['pos'])
            self.samples[qid]['neg'] = list(self.samples[qid]['neg'])
            random.shuffle(self.samples[qid]['neg'])

    def __getitem__(self, item: int) -> Type[InputExample]:
        sample = self.samples[self.sample_ids[item]]
        query_text = sample['query']

        pos_id = sample['pos'].pop(0) # Pop positive and add at end
        pos_text = self.corpus[pos_id]
        sample['pos'].append(pos_id)

        neg_id = sample['neg'].pop(0) # Pop negative and add at end
        neg_text = self.corpus[neg_id]
        sample['neg'].append(neg_id)

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.samples)
