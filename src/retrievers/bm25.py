import os
import json
import time
import pickle
import logging
import argparse
import itertools
from tqdm import tqdm
from os.path import join
from dotenv import load_dotenv
load_dotenv()

import math
import ir_datasets
import numpy as np
import pandas as pd
import seaborn as sns
from statistics import mean
from datasets import load_dataset
from collections import Counter, defaultdict

try:
    from src.utils.metrics import Metrics
except ModuleNotFoundError:
    import sys, pathlib
    sys.path.append(str(pathlib.Path().resolve()))
from src.utils.metrics import Metrics
from src.utils.common import log_step
from src.utils.loggers import LoggingHandler
from src.data.preprocessor import TextPreprocessor


class TFIDF:
    """
    Implementation of the TF-IDF retrieval model.
    """
    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.vocab = self._build_vocab()
        self.tf = self._build_tf_index()
        self.df = self._build_df_index()
        self.idf = self._build_idf_index()

    def __repr__(self):
        return f"{self.__class__.__name__}".lower()

    def get_vocab(self):
        """ Return the vocabulary sorted by alphabeticcal order. """
        return sorted(self.vocab)

    @log_step
    def _build_vocab(self) -> list[str]:
        """ Build the vocabulary from the corpus. """
        return set(word for doc in self.corpus for word in doc.split())

    @log_step
    def _build_tf_index(self, idx_type: int = 2) -> dict[str, dict[int, int]]:
        """ Calculate the term frequency of each word in the vocabulary within each document of the corpus. """
        tf = defaultdict(lambda: defaultdict(int))
        for i, doc in enumerate(self.corpus):
            for word in doc.split():
                if word in self.vocab:
                    tf[word][i] += 1
        return {word: dict(doc_freq) for word, doc_freq in tf.items()}

    @log_step
    def _build_df_index(self) -> dict[str, int]:
        """ Calculate the document frequency of each word in the vocabulary across the corpus. """
        df = Counter()
        for doc in self.corpus:
            df.update(set(doc.split()))
        for word in self.vocab:
            df.setdefault(word, 0)
        return df

    @log_step
    def _build_idf_index(self) -> dict[str, float]:
        """ Calculate the inverse document frequency of each word in the vocabulary across the corpus. """
        idf = dict.fromkeys(self.vocab, 0)
        for word,_ in idf.items():
            idf[word] = self._compute_idf(word)
        return idf

    def _compute_idf(self, word: str) -> float:
        """ Compute the inverse document frequency of a word. """
        return math.log10((self.corpus_size + 1) / (self.df.get(word, 0) + 1))

    @log_step
    def search_all(self, queries: list[str], top_k: int) -> list:
        """ Perform retrieval on all provided queries. """
        results = list()
        t0 = time.perf_counter()
        for q in tqdm(queries, desc='Searching queries'):
            results.append(self.search(q, top_k))
        t1 = time.perf_counter()
        print(f"Avg. latency (ms/quey): {((t1 - t0) / len(queries)) * 1000}")
        return results

    def search(self, query: str, top_k: int) -> list:
        """ Perform retrieval on a single query. """
        results = dict()
        for i,_ in enumerate(self.corpus):
            results[i] = self.score(query, i)
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{'corpus_id': i, 'score': s} for i, s in sorted_results]

    def score(self, query: str, doc_idx: int) -> float:
        """ Compute the TF-IDF score of a query against a document. """
        score = 0.0
        for t in query.split():
            tf = self.tf.get(t, {}).get(doc_idx, 0)
            idf = self.idf.get(t, 0)
            score += tf * idf
        return score

    def save_indexes(self, output_dir: str, dataset: str) -> None:
        """ Save the indexes to disk with pickle. """
        with open(join(output_dir, f'{self.__repr__()}_vocab_{dataset}.pkl'), 'wb') as f:
            pickle.dump(self.vocab, f)
        with open(join(output_dir, f'{self.__repr__()}_tf_{dataset}.pkl'), 'wb') as f:
            pickle.dump(self.tf, f)
        with open(join(output_dir, f'{self.__repr__()}_df_{dataset}.pkl'), 'wb') as f:
            pickle.dump(self.df, f)
        with open(join(output_dir, f'{self.__repr__()}_idf_{dataset}.pkl'), 'wb') as f:
            pickle.dump(self.idf, f)


class BM25(TFIDF):
    """ 
    Implementation of the BM25 retrieval model.
    """
    def __init__(self, corpus: list[str], k1: float, b: float):
        self.b = b
        self.k1 = k1
        super().__init__(corpus)
        self.doc_len = self._build_dl_index()
        self.avgdl = mean(self.doc_len)

    @log_step
    def _build_dl_index(self) -> list[int]:
        """ Calculate the length of each document in the corpus. """
        return [len(doc.split()) for doc in self.corpus]

    def _compute_idf(self, word: str) -> float:
        """ Compute the inverse document frequency of a word. """
        return math.log10((self.corpus_size - self.df.get(word, 0) + 0.5) / (self.df.get(word, 0) + 0.5))

    def score(self, query: str, doc_idx: int) -> float:
        """ Compute the BM25 score of a query against a document. """
        score = 0.0
        for t in query.split():
            tf = self.tf.get(t, {}).get(doc_idx, 0)
            idf = self.idf.get(t, 0)
            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * self.doc_len[doc_idx]/self.avgdl))
        return score

    def update_params(self, k1: float, b: float) -> None:
        """ Update the BM25 parameters. """
        self.k1 = k1
        self.b = b


class AtireBM25(BM25):
    """
    Reference: https://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf
    """
    def __init__(self, corpus: list[str], k1: float, b: float):
        super().__init__(corpus, k1, b)

    def _compute_idf(self, word: str) -> float:
        """ Compute the inverse document frequency of a word. """
        return math.log10((self.corpus_size + 1) / (self.df.get(word, 0) + 1))


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, handlers=[LoggingHandler()])

    logging.info("Loading documents and queries...")
    if args.dataset == 'lleqa':
        dfC = load_dataset('maastrichtlawtech/lleqa', name='corpus', split='corpus', token=os.getenv('HF')).to_pandas()
        corpus = dfC.set_index('id')['article'].to_dict()

        split = 'train' if args.do_negatives_extraction else ('validation' if args.do_hyperparameter_tuning else 'test')
        dfQ = load_dataset('maastrichtlawtech/lleqa', name='questions', split=split, token=os.getenv('HF')).to_pandas()
        qids, queries, pos_pids = dfQ['id'].tolist(), dfQ['question'].tolist(), dfQ['article_ids'].tolist()
    
    elif args.dataset.startswith('mmarco'):
        lang = args.dataset.split('-')[-1]
        dataset = "msmarco-passage" if lang == "en" else f"mmarco/v2/{lang}"
        collection = ir_datasets.load(dataset)
        corpus = {int(d.doc_id): d.text for d in collection.docs_iter()}

        split = 'train' if args.do_negatives_extraction else 'dev/small'
        data = ir_datasets.load(f"{dataset}/{split}")
        qrels, qids, queries, pos_pids = {}, [], [], []
        for s in data.qrels_iter():
            qrels.setdefault(int(s.query_id), []).append(int(s.doc_id))
        for s in data.queries_iter():
            qid = int(s.query_id)
            qids.append(qid)
            queries.append(s.text)
            pos_pids.append(qrels[qid])
    
    documents = list(corpus.values())
    idx2id = {i: pid for i, pid in enumerate(corpus.keys())}

    if args.do_preprocessing:
        logging.info("Preprocessing documents and queries (lemmatizing=True)...")
        cleaner = TextPreprocessor(spacy_model="fr_core_news_md")
        documents = cleaner.preprocess(documents, lemmatize=True)
        queries = cleaner.preprocess(queries, lemmatize=True)

    if args.do_hyperparameter_tuning:
        logging.info("Starting hyperparameter tuning...")
        # Init evaluator and BM25 retriever module.
        evaluator = Metrics(recall_at_k=[10, 100, 200, 500, 1000])
        retriever = BM25(corpus=documents, k1=0., b=0.)

        # Create dataframe to store results.
        hyperparameters = ['k1', 'b']
        metrics = [f"recall@{k}" for k in evaluator.recall_at_k]
        grid_df = pd.DataFrame(columns=hyperparameters+metrics)

        # Create all possible combinations of hyperparamaters.
        k1_range = np.arange(0., 8.5, 0.5)
        b_range = np.arange(0., 1.1, 0.1)
        combinations = list(itertools.product(*[k1_range, b_range]))

        # Launch grid search runs.
        for i, (k1, b) in enumerate(combinations):
            logging.info(f"\n\n({i+1}) Model: BM25 - k1={k1}, b={b}")
            retriever.update_params(k1, b)
            ranked_lists = retriever.search_all(queries, top_k=1000)
            ranked_lists = [[idx2id.get(x['corpus_id']) for x in results] for results in ranked_lists]
            scores = evaluator.compute_all_metrics(all_ground_truths=pos_pids, all_results=ranked_lists)
            scores.update({**{'k1':k1, 'b':b}, **{f"{metric}@{k}": v for metric, results in scores.items() if isinstance(results, dict) for k,v in results.items()}})
            scores.pop('recall')
            grid_df = grid_df.append(scores, ignore_index=True)
            grid_df.to_csv(join(args.output_dir, 'bm25_tuning_results.csv'), sep=',', float_format='%.5f', index=False)
        
        # Plot heatmap.
        grid_df = grid_df.pivot_table(values='recall@100', index='k1', columns='b')[::-1] *100
        plot = sns.heatmap(grid_df, annot=True, cmap="YlOrBr", fmt='.1f', cbar=False, vmin=40, vmax=60)
        plot.get_figure().savefig(join(args.output_dir, "bm25_tuning_heatmap.pdf"))

    else:
        logging.info("Initializing the BM25 retriever model...")
        retriever = BM25(corpus=documents, k1=args.k1, b=args.b)

        logging.info("Running BM25 model on queries...")
        ranked_lists = retriever.search_all(queries, top_k=1000)
        ranked_lists = [[idx2id.get(x['corpus_id']) for x in results] for results in ranked_lists]

        if args.do_evaluation:
            logging.info("Computing the retrieval scores...")
            evaluator = Metrics(recall_at_k=[5, 10, 20, 50, 100, 200, 500, 1000], map_at_k=[10, 100], mrr_at_k=[10, 100], ndcg_at_k=[10, 100])
            scores = evaluator.compute_all_metrics(all_ground_truths=pos_pids, all_results=ranked_lists)
            with open(join(args.output_dir, f'performance_bm25_{args.dataset}_dev.json'), 'w') as f:
                json.dump(scores, f, indent=2)

        if args.do_negatives_extraction:
            logging.info(f"Extracting top-{args.num_negatives} negatives for each question...")
            results = dict()
            for q_id, truths_i, preds_i in zip(qids, pos_pids, ranked_lists):
                results[q_id] = [y for y in preds_i if y not in truths_i][:args.num_negatives]
            results = dict(sorted(results.items()))
            with open(join(args.output_dir, f'negatives_bm25.json'), 'w') as f:
                json.dump(results, f, indent=2)

        retriever.save_indexes(output_dir=args.output_dir, dataset=args.dataset)
    logging.info("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset to use.", choices=[
        "lleqa", "mmarco-ar", "mmarco-de", "mmarco-en", "mmarco-es", "mmarco-fr", "mmarco-hi", "mmarco-id", 
        "mmarco-it", "mmarco-ja", "mmarco-nl", "mmarco-pt", "mmarco-ru", "mmarco-vi", "mmarco-zh",
    ])
    parser.add_argument("--do_preprocessing", action='store_true', default=False, help="Whether to pre-process the articles (lowercasing, lemmatization, and deletion of stopwords, punctuation, and numbers).")
    parser.add_argument("--k1", type=float, default=1.5, help="k1 parameter for the BM25 retrieval model.")
    parser.add_argument("--b", type=float, default=0.75, help="b parameter for the BM25 retrieval model.")
    parser.add_argument("--do_evaluation", action='store_true', default=False, help="Whether to perform evaluation.")
    parser.add_argument("--do_negatives_extraction", action='store_true', default=False, help="Whether to extract top-k BM25 negatives for each question.")
    parser.add_argument("--num_negatives", type=int, default=10, help="Number of negatives to extract per question.")
    parser.add_argument("--do_hyperparameter_tuning", action='store_true', default=False, help="Whether to tune the k1 and b parameters.")
    parser.add_argument("--output_dir", type=str, help="Path of the output directory.")
    args, _ = parser.parse_known_args()
    main(args)
