import os
import csv
import json
import random
from tqdm import tqdm
from os.path import exists, join
from typing import Optional, List, Tuple, Dict, Type
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset
from sentence_transformers import InputExample

try:
    from src.utils.common import read_json_file
except ModuleNotFoundError:
    import sys, pathlib
    sys.path.append(str(pathlib.Path().resolve()))
from src.utils.common import read_json_file


class LLeQADataset(Dataset):
    """ 
    Pytorch Dataset for the LLeQA dataset.

    :param queries: A pandas DataFrame containing the queries.
    :param documents: A pandas DataFrame containing the documents.
    :param stage: The stage of the dataset (either 'train', 'dev', or 'test').
    :param add_doc_title: Whether or not we should append the document title before its content.
    :param hard_negatives: A dictionary containing hard negatives for the queries in the format {qid: [neg1,...,negN]}.
    :param return_samples_as_ids: A boolean that, if set to True, returns the ids of the samples instead of their text sequences.
        (i.e., __getitem__ will return [qid, pos_id, neg_id] instead of [query_text, pos_text, neg_text]).
    :param return_st_format: A boolean that, if set to True, returns the samples inside the InputExample object from the sentence-transformers' library.
    """
    def __init__(self, 
        documents: pd.DataFrame,
        queries: Optional[pd.DataFrame] = None, 
        stage: Optional[str] = None,
        add_doc_title: Optional[bool] = False,
        hard_negatives: Optional[Dict[str, List[str]]] = None,
        return_samples_as_ids: Optional[bool] = False,
        return_st_format: Optional[bool] = False,
    ):
        self.stage = stage
        self.add_doc_title = add_doc_title 
        self.hard_negatives = hard_negatives
        self.return_samples_as_ids = return_samples_as_ids
        self.return_st_format = return_st_format
        self.documents = self.get_id_document_pairs(documents) #Dict[docid, document]
        if queries is not None:
            self.queries = self.get_id_query_pairs(queries) #Dict[qid, query]
            self.one_to_one_pairs = self.get_one_to_one_relevant_pairs(queries) #List[(qid, pos_docid_i)]
            self.one_to_many_pairs =  self.get_one_to_many_relevant_pairs(queries) #Dict[qid, List[pos_docid_i]]

    def __len__(self) -> int:
        if self.stage == "train":
            return len(self.one_to_one_pairs)
        return len(self.one_to_many_pairs)

    def __getitem__(self, idx: int) -> Type[InputExample]:
        pos_id, neg_id, pos_doc, neg_doc = (None, None, None, None)
        if self.stage == "train":
            # Get query and positive document.
            qid, pos_id = self.one_to_one_pairs[idx]
            query = self.queries[qid]
            pos_doc = self.documents[pos_id]
            if self.hard_negatives is not None:
                # Get one hard negative for the query (by poping the first one in the list and adding it back at the end).
                neg_id = self.hard_negatives[str(qid)].pop(0)
                neg_doc = self.documents[neg_id]
                self.hard_negatives[str(qid)].append(neg_id)
                sample = [qid, pos_id, neg_id] if self.return_samples_as_ids else [query, pos_doc, neg_doc]
            else:
                sample = [qid, pos_id] if self.return_samples_as_ids else [query, pos_doc]
        else:
            qid, query = list(self.queries.items())[idx]
            sample = [qid] if self.return_samples_as_ids else [query]
        return InputExample(texts=sample) if self.return_st_format else sample

    def get_id_query_pairs(self, queries: pd.DataFrame) -> Dict[str, str]:
        return queries.set_index('id')['question'].astype('str').to_dict()

    def get_id_document_pairs(self, documents: pd.DataFrame) -> Dict[str, str]:
        if self.add_doc_title:
            documents['article'] = documents['description'].apply(lambda x: x + " | " if len(x) > 0 else None).fillna('') + documents['article']
        return documents.set_index('id')['article'].astype('str').fillna('').to_dict()

    def get_one_to_one_relevant_pairs(self, queries: pd.DataFrame) -> List[Tuple[int, int]]:
        return (queries
            .explode('article_ids')
            .drop(columns=['question', 'answer', 'regions', 'topics'], errors='ignore')
            .drop(columns=queries.filter(regex='paragraph_ids$').columns)
            .rename(columns={'article_ids':'article_id','id':'question_id'})
            .apply(pd.to_numeric)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
            .to_records(index=False)
        )

    def get_one_to_many_relevant_pairs(self, queries: pd.DataFrame) -> Dict[str, List[str]]:
        return queries.set_index('id')['article_ids'].to_dict()



class LLeQACrossencoderLoader:
    """
    Data loader for sampling the LLeQA dataset to train/evaluate cross-encoder models.

    :param load_dev: Whether to load the dev data.
    :param load_test: Whether to load the test data.
    :param load_train: Whether to load the training data.
    :param negatives_system: The system to get the negatives from (either 'bm25' or 'me5').
    :param max_train_examples: Maximum number of training samples to sample. Default to the number of query-pos training pais in LLeQA.
    :param data_folder: str: Folder in which to save the downloaded datasets.
    """
    def __init__(
        self, 
        load_dev: Optional[bool] = False,
        load_test: Optional[bool] = False,
        load_train: Optional[bool] = False,
        negatives_system: Optional[str] = 'me5',
        max_train_examples: Optional[int] = 9330,
        data_folder: Optional[str] = 'data/lleqa',
    ):
        self.load_dev = load_dev
        self.load_test = load_test
        self.load_train = load_train
        self.negatives_system = negatives_system
        self.max_train_examples = max_train_examples
        self.data_folder = data_folder

    def load(self) -> dict:
        articles = self._load_corpus()

        if self.load_dev:
            dev_samples = self._load_eval_samples('dev', articles)

        if self.load_test:
            test_samples = self._load_eval_samples('test', articles)
        
        if self.load_train:
            train_dataset = self._load_train_dataset()
            random.seed(42)
            num_examples = 0
            train_samples = []
            while num_examples < self.max_train_examples:
                for idx in tqdm(range(len(train_dataset)), desc=f"Sampling (qid,pid,rel) tuples -> Pass {(num_examples/2)//len(train_dataset)}"):
                    qid, pos_pid, hard_neg_pid = train_dataset[idx]

                    train_samples.append(InputExample(texts=[train_dataset.queries[qid], train_dataset.documents[pos_pid]], label=1))
                    # if (num_examples % 4) == 0:
                    #     train_samples.append(InputExample(texts=[train_dataset.queries[qid], train_dataset.documents[hard_neg_pid]], label=0))
                    # else:
                    random_neg_pid = random.sample(set(train_dataset.documents.keys()) - set(train_dataset.one_to_many_pairs[qid]), 1)[0]
                    train_samples.append(InputExample(texts=[train_dataset.queries[qid], train_dataset.documents[random_neg_pid]], label=0))

                    num_examples += 2
                    if num_examples >= self.max_train_examples:
                        break

        return {
            'train': train_samples if self.load_train else None,
            'dev': dev_samples if self.load_dev else None,
            'test': test_samples if self.load_test else None,
            'corpus': articles,
        }

    def _load_corpus(self) -> dict[int, str]:
        """
        Loads the corpus of articles from the LLeQA dataset.

        :returns: A dictionary containing the articles in the format {pid: text}.
        """
        corpus = load_dataset('maastrichtlawtech/lleqa', name='corpus', split='corpus', token=os.getenv('HF')).to_pandas()
        corpus_dataset = LLeQADataset(documents=corpus, add_doc_title=False)
        return corpus_dataset.documents

    def _load_eval_samples(self, split: str, articles: dict[int, str], rerank_k_only: int = None) -> dict:
        """ 
        Loads the eval samples for the specified split of the LLeQA dataset.

        :param split: The split to load (either 'dev' or 'test').
        :param articles: A dictionary containing the articles in the format {pid: text}.
        :returns: A dictionary containing the eval samples in the format {qid: {'query': str, 'positive': set[str], 'negative': set[str]}}.
        """
        assert split in ['dev', 'test']
        samples = {}
        random.seed(42)
        dataset = load_dataset('maastrichtlawtech/lleqa', name='questions', token=os.getenv('HF'), split='validation' if split == 'dev' else 'test')
        for s in dataset:
            pos_pids =  s['article_ids']
            neg_pids =  set(articles.keys()) - set(pos_pids)
            if rerank_k_only is not None:
                neg_pids = random.sample(neg_pids, max(0, rerank_k_only - len(pos_pids)))
            samples[s['id']] = {
                "query": s['question'],
                "positive": {articles[pid] for pid in pos_pids},
                "negative": {articles[pid] for pid in neg_pids},
            }
        return samples

    def _load_train_dataset(self) -> LLeQADataset:
        """
        Loads the training dataset for the LLeQA dataset.
        """
        corpus = load_dataset('maastrichtlawtech/lleqa', name='corpus', split='corpus', token=os.getenv('HF')).to_pandas()
        samples = load_dataset('maastrichtlawtech/lleqa', name='questions', split='train', token=os.getenv('HF')).to_pandas()
        negatives = load_dataset('maastrichtlawtech/lleqa', name='negatives', split=self.negatives_system, token=os.getenv('HF')).to_pandas().map(list).to_dict(orient='records')[0]
        return LLeQADataset(queries=samples, documents=corpus, stage='train', hard_negatives=negatives, return_samples_as_ids=True)


class LLeQAColbertLoader:
    """
    Data loader for sampling the LLeQA dataset to train/evaluate ColBERT models.

    :param load_dev: Whether to load the dev data.
    :param load_test: Whether to load the test data.
    :param load_train: Whether to load the training data.
    :param negatives_system: The system to get the negatives from (either 'bm25' or 'me5').
    :param max_train_examples: Maximum number of training samples to sample. Default to the number of query-pos training pais in LLeQA.
    :param data_folder: str: Folder in which to save the created datasets if any.
    """
    def __init__(
        self, 
        load_dev: Optional[bool] = False,
        load_test: Optional[bool] = False,
        load_train: Optional[bool] = False,
        negatives_system: Optional[str] = 'me5',
        max_train_examples: Optional[int] = 9330,
        data_folder: Optional[str] = 'data/lleqa',
    ):
        self.load_dev = load_dev
        self.load_test = load_test
        self.load_train = load_train
        self.negatives_system = negatives_system
        self.max_train_examples = max_train_examples
        self.data_folder = data_folder

    def load(self) -> dict[str, str]:
        data_filepaths = {}

        corpus_filepath = join(self.data_folder, "collection.tsv")
        if not exists(corpus_filepath):
            corpus_dataset = self._load_dataset('corpus')
            self._to_disk(dataset=corpus_dataset, data_type='documents', outpath=corpus_filepath)
        data_filepaths['collection'] = corpus_filepath

        if self.load_dev:
            data_filepaths.update(self._write_data_if_not_exists('dev'))

        if self.load_test:
            data_filepaths.update(self._write_data_if_not_exists('test'))

        if self.load_train:
            train_queries_filepath = join(self.data_folder, f"queries.train.tsv")
            if not exists(train_queries_filepath):
                train_dataset = self._load_dataset('train')
                self._to_disk(dataset=train_dataset, data_type='queries', outpath=train_queries_filepath)
            data_filepaths['train_queries'] = train_queries_filepath

            train_triplets_filepath = join(self.data_folder, f"train-{(self.max_train_examples/1e3):.1f}K-triplets-hard_negs_{self.negatives_system}.jsonl")
            if not exists(train_triplets_filepath):
                train_dataset = self._load_dataset('train')
                num_examples = 0
                with open(train_triplets_filepath, 'w') as fOut:
                    while num_examples < self.max_train_examples:
                        for idx in tqdm(range(len(train_dataset)), desc=f"Sampling (qid,pos_id,neg_id) triplets -> Pass {num_examples//len(train_dataset)}"):
                            sample = [int(x)-1 for x in train_dataset[idx]] # ID-1 as article IDs have been decremented in the saved 'collection.tsv' (see _to_disk)
                            fOut.write(json.dumps(sample) + '\n')
                            num_examples += 1
                            if num_examples >= self.max_train_examples:
                                break
            data_filepaths['train_tuples'] = train_triplets_filepath
        
        return data_filepaths

    def _load_dataset(self, split: str) -> Dataset:
        """
        Loads the specified split of the LLeQA dataset from the Hugging Face Hub.

        :param split: The split to load (either 'corpus', 'dev', 'test', or 'train').
        :returns: A LLeQADataset object containing the data for the specified split.
        """
        assert split in ['corpus', 'dev', 'test', 'train']
        corpus = load_dataset('maastrichtlawtech/lleqa', name='corpus', split='corpus', token=os.getenv('HF')).to_pandas()
        if split == "corpus":
            dataset = LLeQADataset(documents=corpus, add_doc_title=False)
        elif split == "dev":
            samples = load_dataset('maastrichtlawtech/lleqa', name='questions', split='validation', token=os.getenv('HF')).to_pandas()
            dataset = LLeQADataset(queries=samples, documents=corpus, stage='dev')
        elif split == 'test':
            samples = load_dataset('maastrichtlawtech/lleqa', name='questions', split='test', token=os.getenv('HF')).to_pandas()
            dataset = LLeQADataset(queries=samples, documents=corpus, stage='test')
        else:
            samples = load_dataset('maastrichtlawtech/lleqa', name='questions', split='train', token=os.getenv('HF')).to_pandas()
            negatives = load_dataset('maastrichtlawtech/lleqa', name='negatives', split=self.negatives_system, token=os.getenv('HF')).to_pandas().map(list).to_dict(orient='records')[0]
            dataset = LLeQADataset(queries=samples, documents=corpus, stage='train', hard_negatives=negatives, return_samples_as_ids=True)
        return dataset

    def _write_data_if_not_exists(self, split: str) -> dict[str, str]:
        """
        Saves the eval queries and qrels for the specified split if they do not already exist on disk.

        :param split: The split to save the eval queries and qrels for (either 'dev' or 'test').
        :returns: A dictionary containing the paths to the eval queries (in the form "qid \t query") and qrels (in the form: "qid \t 0 pos_pid \t 1").
        """
        dataset = None
        assert split in ['dev', 'test']
        
        queries_filepath = join(self.data_folder, f"queries.{split}.tsv")
        if not exists(queries_filepath):
            dataset = self._load_dataset(split)
            self._to_disk(dataset=dataset, data_type='queries', outpath=queries_filepath)
        
        qrels_filepath = join(self.data_folder, f"qrels.{split}.tsv")
        if not exists(qrels_filepath):
            if dataset is None:
                dataset = self._load_dataset(split)
            self._to_disk(dataset=dataset, data_type='qrels', outpath=qrels_filepath)

        return {
            f'{split}_queries': queries_filepath, 
            f'{split}_qrels': qrels_filepath,
        }
    
    def _to_disk(self, dataset: LLeQADataset, data_type: str, outpath: str) -> None:
        """
        Writes the data from the LLeQADataset object to disk.
        NB: The question and article IDs are decremented to meet the colbert-ir lib's requirements (i.e., corpus_id == line_id).

        :param dataset: The LLeQADataset object to write to disk.
        :param data_type: The type of data to write (either 'queries', 'documents', or 'qrels').
        :param outpath: The path to the file to write the data to.
        """
        assert data_type in ['queries', 'documents', 'qrels']
        with open(outpath, 'w', newline='') as fOut:
            writer = csv.writer(fOut, delimiter='\t')
            if data_type == "qrels":
                for qid, pos_pid in dataset.one_to_one_pairs:
                    writer.writerow([qid-1, 0, pos_pid-1, 1])
            else:
                for text_id, text in getattr(dataset, data_type).items():
                    writer.writerow([text_id-1, text.replace('\n', ' ').replace('\r', ' ')])


class LLeQABiencoderLoader:
    """
    Data loader for sampling the LLeQA dataset to train/evaluate bi-encoder models.

    :param load_train: Whether to load the training data.
    :param load_dev: Whether to load the dev data.
    :param load_test: Whether to load the test data.
    :param negatives_system: The system to get the negatives from (either 'bm25' or 'me5').
    :param synthetic_path_or_url: Path or URL to a JSON file containing synthetic samples to be added to the training set.
    :param synthetic_negatives_path_or_url: Path or URL to a JSON file containing hard negatives for the synthetic samples.
    :param return_st_format: A boolean that, if set to True, returns the samples inside the InputExample object from the sentence-transformers' library.
    """
    def __init__(self,
        load_dev: Optional[bool] = False,
        load_test: Optional[bool] = False,
        load_train: Optional[bool] = False,
        negatives_system: Optional[str] = 'me5',
        synthetic_path_or_url: Optional[str] = None,
        synthetic_negatives_path_or_url: Optional[str] = None,
        return_st_format: Optional[bool] = False,
    ):
        assert load_dev or load_test or load_train, f"All loading modes ('load_dev', 'load_test', 'load_train') are set to False."
        assert negatives_system in ['bm25', 'me5'], f"The system to get the negatives from should be either 'bm25' or 'me5'."
        self.load_dev = load_dev
        self.load_test = load_test
        self.load_train = load_train
        self.negatives_system = negatives_system
        self.synthetic_path_or_url = synthetic_path_or_url
        self.synthetic_negatives_path_or_url = synthetic_negatives_path_or_url
        self.return_st_format = return_st_format

    def load(self) -> dict:
        train, dev, test = {}, {}, {}

        # Load corpus of articles.
        corpus = load_dataset('maastrichtlawtech/lleqa', name='corpus', split='corpus', token=os.getenv('HF')).to_pandas()
        corpus_dataset = LLeQADataset(documents=corpus, add_doc_title=False)

        if self.load_dev:
            dev_samples = load_dataset('maastrichtlawtech/lleqa', name='questions', split='validation', token=os.getenv('HF')).to_pandas()
            dev_dataset = LLeQADataset(queries=dev_samples, documents=corpus, stage='dev')

        if self.load_train:
            train_samples = load_dataset('maastrichtlawtech/lleqa', name='questions', split='train', token=os.getenv('HF')).to_pandas()
            
            # Load hard negatives.
            negatives = load_dataset('maastrichtlawtech/lleqa', name='negatives', split=self.negatives_system, token=os.getenv('HF')).to_pandas().map(list).to_dict(orient='records')[0]

            # Use extra synthetic samples for training, if any.
            if self.synthetic_path_or_url is not None:
                synthetic_samples = pd.read_json(self.synthetic_path_or_url)
                train_samples = pd.concat([train_samples, synthetic_samples], ignore_index=True)

                # Load hard negatives for synthetic samples, if any.
                synthetic_negatives = []
                if self.synthetic_negatives_path_or_url is not None:
                    synthetic_negatives = read_json_file(self.synthetic_negatives_path_or_url)
                    negatives = negatives + synthetic_negatives
                elif negatives is not None:
                    raise ValueError("You must provide hard negatives for synthetic samples.")

            # Make sure that no dev sample is also in the (potentially augmented) train set.
            if self.load_dev:
                dup_questions = train_samples[train_samples['question'].isin(dev_samples['question'])]
                if len(dup_questions) > 0:
                    print(f"Found {len(dup_questions)} questions that appear both in train and dev sets. Removing them from train set...")
                    train_samples.drop(dup_questions.index, inplace=True)

            train_dataset = LLeQADataset(queries=train_samples, documents=corpus, stage='train', hard_negatives=negatives, return_st_format=self.return_st_format)
        
        if self.load_test:
            test_samples = load_dataset('maastrichtlawtech/lleqa', name='questions', split='test', token=os.getenv('HF')).to_pandas()
            test_dataset = LLeQADataset(queries=test_samples, documents=corpus, stage='test')

        return {
            'train': train_dataset if self.load_train else None,
            'dev':  {'queries': dev_dataset.queries, 'labels': dev_dataset.one_to_many_pairs} if self.load_dev else None,
            'test': {'queries': test_dataset.queries, 'labels': test_dataset.one_to_many_pairs} if self.load_test else None,
            'corpus': corpus_dataset.documents,
        }
