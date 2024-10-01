import glob
import tqdm
import random
import ir_datasets
import json, gzip, pickle
from os.path import exists, join, basename

from . import util

__all__ = ["MmarcoReader"]


class MmarcoReader:
    """
    Data reader for sampling the MS MARCO dataset for contrastive learning.

    :param lang: Language in which MS Marco is loaded.
    :param load_dev: Whether to load the small dev set of MS MARCO.
    :param load_train: Whether to load the training data of MS MARCO.
    :param max_train_examples: Maximum number of training samples to sample. The official training set contains 808,731 queries yet only 502,939 have associated labels.
    :param training_sample_format: Type of training samples to use ("triplet", "tuple", "tuple_with_scores"). 
        Triplets are training triples of the form [query, positive, negative].
        Tuples are training tuples of the form [query, positive, negative1, ..., negativeN].
        Tuples with scores are training tuples of the form [query, (positive, pos_score), (negative1, neg_score1), ..., (negativeN, neg_scoreN)].
    :param negs_type: Type of negative samples to use ("original" or "hard"). 
        The former are official MS MARCO training triples with a single BM25 negative.
        The latter are custom training samples obtained by mining hard negatives from dense retrievers.
    :param negs_mining_systems: Comma-separated list of systems used for mining hard negatives (only when negs_type == 'hard').
    :param negs_per_query: Number of hard negatives to sample per query (only when negs_type == 'hard').
    :param ce_score_margin: Margin for the cross-encoder score between negative and positive passages (only when negs_type == 'hard').
    :param data_folder: Folder in which to save the downloaded datasets.
    """
    def __init__(
        self, 
        lang: str,
        load_dev: str = False,
        load_train: str = False,
        max_train_examples: int = 502939,
        training_sample_format: str = 'triplet',
        negs_type: str = 'original',
        negs_mining_systems: str = 'all',
        negs_per_query: int = 1,
        ce_score_margin: float = 3.0,
        data_folder: str = 'data/mmarco',
    ):
        self.supported_languages = {
            'ar': ('arabic', 'ar_AR'),
            'de': ('german', 'de_DE'),
            'en': ('english', 'en_XX'),
            'es': ('spanish', 'es_XX'),
            'fr': ('french', 'fr_XX'),
            'hi': ('hindi', 'hi_IN'),
            'id': ('indonesian', 'id_ID'),
            'it': ('italian', 'it_IT'),
            'ja': ('japanese', 'ja_XX'),
            'nl': ('dutch', 'nl_XX'),
            'pt': ('portuguese', 'pt_XX'),
            'ru': ('russian', 'ru_RU'),
            'vi': ('vietnamese', 'vi_VN'),
            'zh': ('chinese', 'zh_CN'),
        }
        self.supported_negative_mining_systems = {
            # https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives
            'bm25': 1, 
            'msmarco-distilbert-base-tas-b': 2,
            'msmarco-distilbert-base-v3': 3,
            'msmarco-MiniLM-L-6-v3': 4,
            'distilbert-margin_mse-cls-dot-v2': 5,
            'distilbert-margin_mse-cls-dot-v1': 6,
            'distilbert-margin_mse-mean-dot-v1': 7,
            'mpnet-margin_mse-mean-v1': 8,
            'co-condenser-margin_mse-cls-v1': 9,
            'distilbert-margin_mse-mnrl-mean-v1': 10,
            'distilbert-margin_mse-sym_mnrl-mean-v1': 11,
            'distilbert-margin_mse-sym_mnrl-mean-v2': 12,
            'co-condenser-margin_mse-sym_mnrl-mean-v1': 13,
        }
        assert lang in self.supported_languages.keys(), (
            f"Language {lang} not supported. Please choose between: {', '.join(self.supported_languages.keys())}."
        )
        assert training_sample_format in ['triplet', 'tuple', 'tuple_with_scores'], (
            f"Unkwown type of training samples {training_sample_format}. Please choose between 'triplet', 'tuple', and 'tuple_with_scores'."
        )
        assert negs_type in ['original', 'hard'], (
            f"Unkwown type of training samples {negs_type}. Please choose between 'original' and 'hard'."
        )
        if negs_type == 'hard':
            assert negs_mining_systems == 'all' or all(syst in self.supported_negative_mining_systems.keys() for syst in negs_mining_systems.split(',')), (
                f"Unsupported negative mining systems. Please choose 'all' or a subset of: {', '.join(self.supported_negative_mining_systems.keys())}."
            )
        self.lang = lang.replace('nl', 'dt')
        self.load_dev = load_dev
        self.load_train = load_train
        self.max_train_examples = max_train_examples
        self.training_sample_format = training_sample_format
        self.negs_type = negs_type
        self.negs_mining_systems = negs_mining_systems.split(',') if negs_mining_systems != 'all' else self.supported_negative_mining_systems.keys()
        self.negs_per_query = negs_per_query
        self.ce_score_margin = ce_score_margin
        self.data_folder = data_folder
    
    def load(self):
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
            # Load training queries.
            train = ir_datasets.load(f"{collection}/train")
            train_queries = {int(q.query_id): q.text for q in train.queries_iter()}

            # Get training sample file path.
            train_filepath = self._get_training_filepath(filename_pattern=(
                f"train-*M" +
                f"-{self.training_sample_format}" +
                f"{'' if self.training_sample_format == 'triplet' else ('-*nway')}" +
                f"-{self.negs_type}_negs" +
                f"{'' if self.negs_type == 'original' else ('-systems_' + '.'.join(map(str, sorted([self.supported_negative_mining_systems.get(syst) for syst in self.negs_mining_systems]))))}" +
                f".jsonl"
            ), target_num_examples=round(self.max_train_examples/1e6, 1), target_nway=self.negs_per_query+1,
            )

            # Load training samples if they exist.
            if exists(train_filepath):
                num_examples = 0
                train_samples = []
                with open(train_filepath, 'r') as fIn:
                    for line in tqdm.tqdm(fIn, desc=f"Loading training samples from {train_filepath}"):
                        if self.training_sample_format == 'triplet':
                            qid, pos_pid, neg_pid = json.loads(line)
                            train_samples.append([train_queries[qid], passages[pos_pid], passages[neg_pid]])
                        elif self.training_sample_format == 'tuple':
                            data = json.loads(line)
                            qid, pids = data[0], data[1:self.negs_per_query + 2]
                            train_samples.append([train_queries[qid]] + [passages[pid] for pid in pids])
                        else:
                            data = json.loads(line)
                            qid, pids_tuples = data[0], data[1:self.negs_per_query + 2]
                            train_samples.append([train_queries[qid]] + [[passages[pid], score] for pid, score in pids_tuples])
                        
                        num_examples += 1
                        if num_examples >= self.max_train_examples:
                            break
            else:
                # Use original triplet samples with BM25 negatives from the official dataset.
                if self.negs_type == "original":
                    if self.training_sample_format != "triplet":
                        raise ValueError(f"Training sample format {self.training_sample_format} not supported for original MS MARCO samples.")
                    url = 'https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz'
                    samples_filepath = util.download_if_not_exists(data_dir=self.data_folder, file_url=url)

                    num_examples = 0
                    train_samples = []
                    with gzip.open(samples_filepath, 'rt') as fIn, open(train_filepath, 'w') as fOut:
                        for line in tqdm.tqdm(fIn, unit_scale=True):
                            qid, pos_pid, neg_pid = [int(x) for x in line.strip().split()]
                            train_samples.append([train_queries[qid], passages[pos_pid], passages[neg_pid]])
                            fOut.write(json.dumps([qid, pos_pid, neg_pid]) + '\n')

                            num_examples += 1
                            if num_examples >= self.max_train_examples:
                                break
                else:
                    # Load CE scores for query-passage pairs: ce_scores[qid][pid] -> score.
                    url = 'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz'
                    ce_scores_file = util.download_if_not_exists(data_dir=self.data_folder, file_url=url)
                    with gzip.open(ce_scores_file, 'rb') as fIn:
                        ce_scores = pickle.load(fIn)

                    # Use hard negatives mined from BM25 + 12 different dense retrievers.
                    url = 'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz'
                    hard_negatives_filepath = util.download_if_not_exists(data_dir=self.data_folder, file_url=url)

                    i, num_examples = 0, 0
                    train_samples = []
                    with gzip.open(hard_negatives_filepath, 'rt') as fIn, open(train_filepath, 'w') as fOut:
                        while num_examples < self.max_train_examples:
                            i += 1
                            fIn.seek(0)
                            random.seed(num_examples)
                            for line in tqdm.tqdm(fIn, desc=f"Sampling training {self.training_sample_format} -> Dataset pass {i}"):
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
                                
                                # Sample N hard negatives and their CE scores.
                                neg_pids = []
                                systems = [syst for syst in data['neg'].keys() if syst in self.negs_mining_systems]
                                for system_name in systems:
                                    neg_pids.extend(data['neg'][system_name])
                                filtered_neg_pids = [pid for pid in list(set(neg_pids)) if ce_scores[qid][pid] <= ce_score_threshold]
                                sampled_neg_pids = random.sample(filtered_neg_pids, min(self.negs_per_query, len(filtered_neg_pids)))
                                sampled_neg_scores = [ce_scores[qid][pid] for pid in sampled_neg_pids]

                                if len(sampled_neg_pids) == self.negs_per_query:
                                    query = train_queries[qid]
                                    sampled_pos = passages[sampled_pos_pid]
                                    sampled_negs = [passages[pid] for pid in sampled_neg_pids]
                                    if self.training_sample_format == 'triplet':
                                        sample = [query, sampled_pos, sampled_negs[0]]
                                        qrel = [qid, sampled_pos_pid, sampled_neg_pids[0]]
                                    elif self.training_sample_format == 'tuple':
                                        sample = [query, sampled_pos] + sampled_negs
                                        qrel = [qid] + [sampled_pos_pid] + sampled_neg_pids
                                    else:
                                        sample = [query, [sampled_pos, sampled_pos_score]] + [list(pair) for pair in zip(sampled_negs, sampled_neg_scores)]
                                        qrel = [qid, [sampled_pos_pid, sampled_pos_score]] + [list(pair) for pair in zip(sampled_neg_pids, sampled_neg_scores)]

                                    train_samples.append(sample)
                                    fOut.write(json.dumps(qrel) + '\n')

                                    num_examples += 1
                                    if num_examples >= self.max_train_examples:
                                        break
                        
                        del ce_scores
                        print(f"#> Number of training examples created: {num_examples}.")
                    
        return {
            'train': train_samples if self.load_train else None,
            'dev': {'queries': dev_queries, 'labels': dev_qrels} if self.load_dev else None,
            'corpus': passages,
        }

    def _get_training_filepath(self, filename_pattern: str, target_num_examples: float, target_nway: int = None):
        """
        Get the file path of the training samples with the target number of examples.

        :param filename_pattern: Pattern of the training sample file name.
        :param target_num_examples: Target number of training examples.
        :returns: File path of the training samples with the target number of examples.
        """
        for filepath in glob.glob(join(self.data_folder, filename_pattern)):
            num_samples = float(basename(filepath).split('-')[1][:-1])
            nway = int(basename(filepath).split('nway')[0].split('-')[-1]) if 'nway' in basename(filepath) else None
            if nway is None or (nway is not None and nway >= target_nway):
                if num_samples >= target_num_examples:
                    return filepath
        new_filename = filename_pattern.replace('*M', f'{target_num_examples:.1f}M').replace('*nway', f'{target_nway}nway')
        return join(self.data_folder, new_filename)
