import os
import sys
import copy
import pathlib
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from os.path import exists, join
from collections import defaultdict, OrderedDict

import math
import torch
import random
import itertools
import numpy as np
import pandas as pd
from datasets import load_dataset
from torch.cuda import empty_cache

load_dotenv()
sys.path.append(str(pathlib.Path().resolve()))


def run_evaluation(predictions: list[list[int]], labels: list[list[int]], print2console: bool = True, log2wandb: bool = False, args: argparse.Namespace = None):
    from src.utils.metrics import Metrics
    from src.utils.loggers import WandbLogger

    evaluator = Metrics(recall_at_k=[5,10,20,50,100,200,500,1000], map_at_k=[10,100], mrr_at_k=[10,100], ndcg_at_k=[10,100])
    scores = evaluator.compute_all_metrics(all_ground_truths=labels, all_results=predictions)
    if print2console:
        for metric, score in scores.items():
            print(f'- {metric.capitalize()}: {score:.3f}')
    if log2wandb:
        logger = WandbLogger(
            project_name="lleqa", 
            run_name=f"hybrid-{args.eval_type}-{args.fusion}-{args.normalization+'-' if args.fusion == 'nsf' else ''}{'-'.join([n.split('_')[-1] for n, val in vars(args).items() if n.startswith('run_') and val is True])}",
            run_config=args, 
            log_dir=join(args.output_dir, 'logs'),
        )
        for metric, score in scores.items():
            logger.log_eval(0, 0, f'{args.data_split}/{metric}', score)
    return scores


class Ranker:
    """
    A class for ranking queries against a corpus.
    """
    @staticmethod
    def bm25_search(queries: list[str], corpus: dict[int, str], do_preprocessing: bool, k1: float, b: float, return_topk: int = None):
        """
        Perform retrieval on a corpus using the BM25 algorithm.

        :param queries: A list of queries to rank.
        :param corpus: A dictionary of "id: article" items.
        :param do_preprocessing: Whether to preprocess the articles before searching.
        :param k1: The k1 parameter for the BM25 algorithm.
        :param b: The b parameter for the BM25 algorithm.
        :param return_topk: The number of results to return.
        :return: A list of dictionaries containing the ranked lists.
        """
        from src.retrievers.bm25 import BM25
        from src.data.preprocessor import TextPreprocessor

        documents = list(corpus.values())
        idx2id = {i: pid for i, pid in enumerate(corpus.keys())}

        if do_preprocessing:
            cleaner = TextPreprocessor(spacy_model="fr_core_news_md")
            documents = cleaner.preprocess(documents, lemmatize=True)
            queries = cleaner.preprocess(queries, lemmatize=True)
        
        retriever = BM25(corpus=documents, k1=k1, b=b)
        ranked_lists = retriever.search_all(queries, top_k=return_topk or len(documents))
        return [[{'corpus_id': idx2id.get(x['corpus_id']), 'score': x['score']} for x in res] for res in ranked_lists]

    @staticmethod
    def single_vector_search(queries: list[str], corpus: dict[int, str], model_name_or_path: str, return_topk: int = None):
        """
        Perform retrieval on a corpus using a single vector search model.

        :param queries: A list of queries to rank.
        :param corpus: A dictionary of "id: article" items.
        :param model_name_or_path: The name of the model to use.
        :param return_topk: The number of results to return.
        :return: A list of dictionaries containing the ranked lists.
        """
        from sentence_transformers import util
        from src._temp.neural_cherche.models import SPLADE
        from src.utils.sentence_transformers import SentenceTransformerCustom
        
        documents = list(corpus.values())
        idx2id = {i: pid for i, pid in enumerate(corpus.keys())}

        if 'splade' in model_name_or_path.lower():
            model = SPLADE(model_name_or_path, max_query_length=64, max_doc_length=512)
        else:
            model = SentenceTransformerCustom(model_name_or_path)
            model.max_seq_length = 512
        
        d_embs = model.encode(sentences=documents, batch_size=64, convert_to_tensor=True, show_progress_bar=True, **({'query_mode': False} if isinstance(model, SPLADE) else {}))
        q_embs = model.encode(sentences=queries, batch_size=64, convert_to_tensor=True, show_progress_bar=True, **({'query_mode': True} if isinstance(model, SPLADE) else {}))
        ranked_lists = util.semantic_search(query_embeddings=q_embs, corpus_embeddings=d_embs, top_k=return_topk or len(documents), score_function=util.cos_sim)

        del model, d_embs, q_embs; empty_cache()
        return [[{'corpus_id': idx2id.get(x['corpus_id']), 'score': x['score']} for x in res] for res in ranked_lists]

    @staticmethod
    def multi_vector_search(queries: list[str], corpus: dict[int, str], model_name_or_path: str, output_dir: str = 'output', return_topk: int = None):
        """
        Perform retrieval on a corpus using a multi-vector search model.

        :param queries: A list of queries to rank.
        :param corpus: A dictionary of "id: article" items.
        :param model_name_or_path: The name of the model to use.
        :param output_dir: The output directory to use.
        :param return_topk: The number of results to return.
        :return: A list of dictionaries containing the ranked lists.
        """
        from colbert import Indexer, Searcher
        from colbert.infra import Run, RunConfig, ColBERTConfig

        documents = list(corpus.values())
        idx2id = {i: pid for i, pid in enumerate(corpus.keys())}

        index_root = f"output/testing/indexes/{model_name_or_path.split('/')[-1]}"
        if not exists(join(index_root, "lleqa.index")):
            with Run().context(RunConfig(nranks=1, index_root=index_root)):
                indexer = Indexer(checkpoint=model_name_or_path, config=ColBERTConfig(query_maxlen=64, doc_maxlen=512))
                indexer.index(name="lleqa.index", collection=documents, overwrite='reuse')

        with Run().context(RunConfig(nranks=1, index_root=index_root)):
            searcher = Searcher(index="lleqa.index", config=ColBERTConfig(query_maxlen=64, doc_maxlen=512))
            ranked_lists = searcher.search_all({idx: query for idx, query in enumerate(queries)}, k=return_topk or len(documents))

        empty_cache()
        return [[{'corpus_id': idx2id.get(x[0]), 'score': x[2]} for x in res] for res in ranked_lists.todict().values()]

    @staticmethod
    def cross_encoder_search(queries: list[str], candidates: list[dict[int, str]], model_name_or_path: str, return_topk: int = None):
        """
        Perform retrieval on a corpus using a cross-encoder model.

        :param queries: A list of queries to rank.
        :param candidates: A list of dictionaries of "id: article" items.
        :param model_name_or_path: The name of the model to use.
        :param return_topk: The number of results to return.
        :return: A list of dictionaries containing the ranked lists.
        """
        from src.utils.sentence_transformers import CrossEncoderCustom

        model = CrossEncoderCustom(model_name_or_path)

        ranked_lists = []
        for query, cands in zip(queries, candidates):
            documents = cands.values()
            idx2id = {i: pid for i, pid in enumerate(cands.keys())}

            results = model.rank(query, documents=docs, top_k=return_topk or len(documents), batch_size=64, show_progress_bar=True)
            ranked_lists.append([{'corpus_id': idx2id.get(x['corpus_id']), 'score': x['score']} for x in results])
        
        del model; empty_cache()
        return ranked_lists


class Aggregator:
    """
    A class for aggregating ranked lists.
    """
    @classmethod
    def fuse(
        cls, 
        ranked_lists: dict[str, list[list[dict]]],
        method: str,
        normalization: str = None, 
        linear_weights: dict[str, float] = None, 
        percentile_distributions: dict[str, np.array] = None,
        return_topk: int = 1000,
    ) -> list[dict[int, float]]:
        """
        Fuse the ranked lists of different retrieval systems.

        :param ranked_lists: A dictionary of ranked lists, where each key is a retrieval system and each value is a list of ranked lists.
        :param method: The fusion method to use.
        :param normalization: The normalization method to use.
        :param linear_weights: A dictionary of weights to use for linear combination.
        :param percentile_distributions: A dictionary of percentile distributions to use for percentile rank normalization.
        :param return_topk: The number of results to return.
        :return: A list of dictionaries containing the final ranked lists.
        """
        num_queries = len(next(iter(ranked_lists.values())))
        assert all(len(system_res) == num_queries for system_res in ranked_lists.values()), (
            "Ranked results from different retrieval systems have varying lenghts across systems (i.e., some systems have been run on more queries)."
        )
        assert (set(ranked_lists.keys()) == set(linear_weights.keys()), (
            "The system names in the provided 'linear_weights' dictionary for convex combination do not correspond to those from the 'ranked_lists'."
        )) if method == 'nsf' else True

        final_results = []
        for i in range(num_queries):
            query_results = []

            for system, results in ranked_lists.items():
                res = cls.convert2dict(results[i])

                if method == 'bcf':
                    res = cls.transform_scores(res, transformation='borda-count')
                
                elif method == 'rrf':
                    res = cls.transform_scores(res, transformation='reciprocal-rank')
                
                elif method == 'nsf':
                    res = cls.transform_scores(res, transformation=normalization, percentile_distr=percentile_distributions.get(system))
                    res = cls.weight_scores(res, w=linear_weights[system])
                
                query_results.append(res)
            
            final_results.append(cls.aggregate_scores(*query_results))
        
        return final_results[:return_topk]

    @staticmethod
    def convert2dict(results: list[dict[str, int|float]]) -> dict[int, float]:
        """
        Convert a list of "result_id: score" dictionaries to a dictionary of "result_id: score" items.

        :param results: A list of "result_id: score" dictionaries.
        :return: A dictionary of "result_id: score" items.
        """
        if sys.version_info >= (3, 7):
            return {res['corpus_id']: res['score'] for res in results}
        else:
            return OrderedDict((res['corpus_id'], res['score']) for res in results)

    @staticmethod
    def transform_scores(results: dict[int, float], transformation: str, percentile_distr: np.array = None) -> dict[int, float]:
        """
        Transform the scores of a list of results.

        :param results: A dictionary of "result_id: score" items.
        :param transformation: The transformation method to use.
        :param percentile_distr: The percentile distribution to use for percentile rank normalization.
        :return: A dictionary of "result_id: score" items.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if transformation == 'borda-count':
            num_candidates = len(results)
            return {pid: (num_candidates - idx+1) / num_candidates for idx, pid in enumerate(results.keys())}
        
        elif transformation == 'reciprocal-rank':
            return {pid: 1 / (60 + idx+1) for idx, pid in enumerate(results.keys())}

        elif transformation == 'min-max':
            scores = torch.tensor(list(results.values()), device=device, dtype=torch.float32)
            min_val, max_val = torch.min(scores), torch.max(scores)
            scores = (scores - min_val) / (max_val - min_val) if min_val != max_val else torch.ones_like(scores)
            return {pid: score for pid, score in zip(results.keys(), scores.cpu().numpy())}

        elif transformation == 'z-score':
            scores = torch.tensor(list(results.values()), device=device, dtype=torch.float32)
            mean_val, std_val = torch.mean(scores), torch.std(scores)
            scores = (scores - mean_val) / std_val if std_val != 0 else torch.zeros_like(scores)
            return {pid: score for pid, score in zip(results.keys(), scores.cpu().numpy())}
        
        elif transformation == 'arctan':
            scores = torch.tensor(list(results.values()), device=device, dtype=torch.float32)
            scores = (2 / math.pi) * torch.atan(0.1 * scores)
            return {pid: score for pid, score in zip(results.keys(), scores.cpu().numpy())}
    
        elif transformation == 'percentile-rank' or transformation == 'normal-curve-equivalent':
            scores = torch.tensor(list(results.values()), device=device, dtype=torch.float32)
            distribution = torch.tensor(percentile_distr, device=device, dtype=torch.float32)
            differences = torch.abs(distribution[:, None] - scores)
            scores = torch.argmin(differences, axis=0) / distribution.size(0)
            if transformation == 'normal-curve-equivalent':
                scores = torch.distributions.Normal(0, 1).icdf(scores / 100) * 21.06 + 50
            return {pid: score for pid, score in zip(results.keys(), scores.cpu().numpy())}
    
        return results

    @staticmethod
    def weight_scores(results: dict[int, float], w: float) -> dict[int, float]:
        """
        Weight the scores of a list of results.

        :param results: A dictionary of "result_id: score" items.
        :param w: The weight to apply.
        :return: A dictionary of "result_id: weighted_score" items.
        """
        return {corpus_id: score * w for corpus_id, score in results.items()}

    @staticmethod
    def aggregate_scores(*args: dict[int, float]) -> list[dict[str, int|float]]:
        """
        Aggregate the scores of a list of lists of results.

        :param args: A list of dictionaries of "result_id: score" items.
        :return: A list of dictionaries of "result_id: score" items.
        """
        agg_results = defaultdict(float)
        for results in args:
            for pid, score in results.items():
                agg_results[pid] += score
        
        agg_results = sorted(agg_results.items(), key=lambda x: x[1], reverse=True)
        return [{'corpus_id': pid, 'score': score} for pid, score in agg_results]


def main(args):
    sep = f"#{'-'*40}#"
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    model_ckpts = {
        'dpr': {
            'general': 'antoinelouis/biencoder-camembert-base-mmarcoFR',
            'legal': 'maastrichtlawtech/dpr-legal-french'
        },
        'splade': {
            'general': 'antoinelouis/spladev2-camembert-base-mmarcoFR',
            'legal': 'maastrichtlawtech/splade-legal-french'
        },
        'colbert': {
            'general': 'antoinelouis/colbertv1-camembert-base-mmarcoFR',
            'legal': 'maastrichtlawtech/colbert-legal-french'
        },
        'monobert': {
            'general': 'antoinelouis/crossencoder-camembert-base-mmarcoFR',
            'legal': 'maastrichtlawtech/monobert-legal-french'
        }
    }
    args.eval_type = ('in' if args.models_domain == 'legal' else 'out') + 'domain'

    print("Loading corpus and queries...")
    dfC = load_dataset('maastrichtlawtech/lleqa', name='corpus', split='corpus', token=os.getenv('HF')).to_pandas()
    corpus = dfC.set_index('id')['article'].to_dict()
    dfQ = load_dataset('maastrichtlawtech/lleqa', name='questions', split='validation' if args.data_split == 'dev' else args.data_split, token=os.getenv('HF')).to_pandas()
    qids, queries, pos_pids = dfQ['id'].tolist(), dfQ['question'].tolist(), dfQ['article_ids'].tolist()

    #------------------------------------------------------#
    #                       RETRIEVAL
    #------------------------------------------------------#
    if args.run_bm25:
        print(f"{sep}\n# Ranking with BM25\n{sep}")
        results['bm25'] = Ranker.bm25_search(queries, corpus, do_preprocessing=True, k1=2.5, b=0.2)

    if args.run_dpr:
        print(f"{sep}\n# Ranking with DPR\n{sep}")
        results['dpr'] = Ranker.single_vector_search(queries, corpus, model_name_or_path=model_ckpts['dpr'][args.models_domain])

    if args.run_splade:
        print(f"{sep}\n# Ranking with SPLADE\n{sep}")
        results['splade'] = Ranker.single_vector_search(queries, corpus, model_name_or_path=model_ckpts['splade'][args.models_domain])

    if args.run_colbert:
        print(f"{sep}\n# Ranking with ColBERT\n{sep}")
        results['colbert'] = Ranker.multi_vector_search(queries, corpus, model_name_or_path=model_ckpts['colbert'][args.models_domain])

    #------------------------------------------------------#
    #                       ANALYSES
    #------------------------------------------------------#
    if args.analyze_score_distributions:
        print(f"{sep}\n# Analyzing the score distributions per system\n{sep}")
        all_scores, labeled_scores = [], []
        
        random.seed(42)
        max_pid = dfC['id'].max()
        neg_pids = dfQ['article_ids'].apply(lambda x: random.sample(list(set(range(1, max_pid+1)) - set(x)), k=len(x))).tolist() # Sample as many random negatives as positives per query.
        
        for i, query in tqdm(enumerate(queries), total=len(queries), desc='Queries'):
            transformed_results, distributions = {}, {}
            if args.normalization == 'percentile-rank' or args.normalization == 'normal-curve-equivalent':
                distr_df = pd.read_csv(join(args.output_dir, f"score_distributions_raw_{args.eval_type}_10k.csv"))
                distributions = {k: np.array(v) for k, v in distr_df.to_dict('series').items()}
            
            for system, res in results.items():
                transformed_results[system] = Aggregator.transform_scores(results=Aggregator.convert2dict(res[i]), transformation=args.normalization, percentile_distr=distributions.get(system))
                all_scores.extend([{"system": system, "score": score} for score in transformed_results[system].values()]) # This list will contain num_queries*corpus_size elements for analyzing the overall score distribution per system.
            
            for label, pids in [("positive", pos_pids[i]), ("negative", neg_pids[i])]:
                for pid in pids:
                    labeled_scores.append({"label": label, **{system: res.get(pid, 0) for system, res in transformed_results.items()}}) # This list will contain num_queries*num_pos_per_query*2 elements for visualizing the scores associated to positives and negatives per system.

        # Save the transformed scores of each system for further analysis.
        all_scores_df = pd.DataFrame(all_scores, columns=list(all_scores[0].keys()))
        all_scores_df.to_csv(join(args.output_dir, f"scores_{args.normalization}_{args.eval_type}_{args.data_split}.csv"), index=False)

        # Convert the complete score distribution of each system into a smaller percentile-based distribution of N data points.
        for N in [1000, 10000, 100000, len(corpus)]:
            (all_scores_df
                .groupby('system').apply(lambda group: group[(group['score'] != 0.0) & (~group['score'].isin(group['score'].drop_duplicates().nsmallest(2)))])
                .reset_index(drop=True)
                .groupby('system').apply(lambda group: pd.Series(group['score'].quantile(np.linspace(0, 1, N+1))))
                .transpose()
                .to_csv(join(args.output_dir, f"score_distributions_{args.normalization}_{args.eval_type}_{round(N/1e3)}k.csv"), index=False)
            )

        # Save the transformed scores of each system together with their pos/neg labels for further analysis.
        labeled_scores_df = pd.DataFrame(labeled_scores, columns=list(labeled_scores[0].keys()))
        labeled_scores_df.to_csv(join(args.output_dir, f"labeled_scores_{args.normalization}_{args.eval_type}_{args.data_split}.csv"), index=False)
        print("Done."); return

    if args.fusion == 'nsf' and args.tune_linear_fusion_weight:
        step = 0.05
        weight_combinations = [
            {name: weight for name, weight in zip(results.keys(), comb)}
            for comb in itertools.product(np.arange(0, 1+step, step), repeat=len(results)) if np.isclose(sum(comb), 1.0)
        ]
        distributions = {}
        if args.normalization == 'percentile-rank' or args.normalization == 'normal-curve-equivalent':
            distr_df = pd.read_csv(join(args.output_dir, f"score_distributions_raw_{args.eval_type}_28k.csv"))
            distributions = {k: np.array(v) for k, v in distr_df.to_dict('series').items()}

        print(f"{sep}\n# Tuning the weights of convex combination between systems: {len(weight_combinations)} permutations\n{sep}")
        all_perf_df = None
        for weights in tqdm(weight_combinations):
            ranked_lists = Aggregator.fuse(copy.deepcopy(results), method=args.fusion, normalization=args.normalization, percentile_distributions=distributions, linear_weights=weights)
            perf_scores = run_evaluation(predictions=[[x['corpus_id'] for x in res] for res in ranked_lists], labels=pos_pids, print2console=False)
            curr_perf_df = (pd.DataFrame([perf_scores])
                .pipe(lambda df: df.assign(**{f'weight_{key}': value for key, value in weights.items()}))
                .pipe(lambda df: df.reset_index(drop=True))
            )
            all_perf_df = curr_perf_df if all_perf_df is None else pd.concat([all_perf_df, curr_perf_df], axis=0)
            all_perf_df.to_csv(join(args.output_dir, f"nsf_{args.normalization}_{args.eval_type}.csv"), index=False)
        print("Done."); return

    #------------------------------------------------------#
    #                       FUSION
    #------------------------------------------------------#
    distributions, weights = {}, {}
    if args.fusion == 'nsf':
        # # Get optimal weights for the current combination.
        # df = pd.read_csv(f'./output/testing/nsf_{args.normalization}_{args.eval_type}.csv')
        # filter_condition = " & ".join(
        #     f"(df.weight_{model} {'!=' if getattr(args, f'run_{model}') else '=='} 0.0)" 
        #     for model in ['bm25', 'splade', 'dpr', 'colbert']
        # )
        # weights = (df[eval(filter_condition)]
        #     .sort_values(by='recall@10', ascending=False)
        #     .iloc[0]
        #     .filter(regex='^weight_')
        #     .rename(lambda x: x.replace('weight_', ''))
        #     .to_dict()
        # )

        # Set weights to equal values.
        weights = {system: 1/len(results) for system in results}

        if args.normalization == 'percentile-rank' or args.normalization == 'normal-curve-equivalent':
            distr_df = pd.read_csv(join(args.output_dir, f"score_distributions_raw_{args.eval_type}_28k.csv"))
            distributions = {k: np.array(v) for k, v in distr_df.to_dict('series').items()}

    print(f"{sep}\n# Fusing results with {args.fusion.upper()}{' ('+args.normalization+')' if args.fusion == 'nsf' else ''}\n{sep}")
    ranked_lists = Aggregator.fuse(ranked_lists=results, method=args.fusion, normalization=args.normalization, percentile_distributions=distributions, linear_weights=weights)
    
    #------------------------------------------------------#
    #                       RERANKING
    #------------------------------------------------------#
    if args.run_monobert:
        print(f"{sep}\n# Re-ranking with monoBERT \n{sep}")
        ranked_lists = Ranker.cross_encoder_search(queries, ranked_lists, model_name_or_path=model_ckpts['monobert'][args.models_domain])

    #------------------------------------------------------#
    #                       EVALUATION
    #------------------------------------------------------#
    print(f"{sep}\n# Evaluation \n{sep}")
    run_evaluation(predictions=[[x['corpus_id'] for x in res] for res in ranked_lists], labels=pos_pids, log2wandb=True, args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split", type=str, help="The LLeQA data split to use.", choices=['dev', 'test', 'train'])
    parser.add_argument("--models_domain", type=str, help="The training domain of the neural retrievers to use.", choices=['general', 'legal'])
    parser.add_argument("--run_bm25", action='store_true', default=False, help="Whether to run BM25 retrieval.")
    parser.add_argument("--run_dpr", action='store_true', default=False, help="Whether to run DPR retrieval.")
    parser.add_argument("--run_splade", action='store_true', default=False, help="Whether to run SPLADE retrieval.")
    parser.add_argument("--run_colbert", action='store_true', default=False, help="Whether to run ColBERT retrieval.")
    parser.add_argument("--run_monobert", action='store_true', default=False, help="Whether to run monoBERT retrieval.")
    parser.add_argument("--fusion", type=str, help="The technique to fuse the ranked lists of of results.", choices=['bcf', 'rrf', 'nsf'])
    parser.add_argument("--normalization", type=str, help="The normalization method to scale the systems' scores (used only when --fusion='nsf'.", choices=[
        'none', 'min-max', 'z-score', 'arctan', 'percentile-rank', 'normal-curve-equivalent'
    ])
    parser.add_argument("--tune_linear_fusion_weight", action='store_true', default=False, help="Whether to tune of the weight parameter when linearly interpolating lexical and dense scores.")
    parser.add_argument("--analyze_score_distributions", action='store_true', default=False, help="Whether to plot the distributions of lexical and dense scores for positive and negative documents.")
    parser.add_argument("--output_dir", type=str, help="Path of the output directory.")
    args, _ = parser.parse_known_args()
    main(args)
