import sys
import torch
import pathlib
import argparse
from os.path import join
from copy import deepcopy
from datetime import datetime

from colbert.data import Queries, Collection
from colbert.infra import Run, RunConfig, ColBERTConfig

try:
    from src.data.mmarco import MmarcoColbertLoader
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.data.mmarco import MmarcoColbertLoader
from src.data.mrtydi import MrTydiColbertLoader
from src.data.lleqa import LLeQAColbertLoader
from src.utils.common import set_seed
from src.utils.colbert_ir import CustomTrainer, CustomIndexer, CustomSearcher, evaluate


def main(args):
    if args.dataset.startswith('mmarco'):
        dataset_name, language = args.dataset.split('-', 1)
        dataloader = MmarcoColbertLoader(
            lang=language,
            load_dev=args.do_test,
            load_train=args.do_train,
            max_train_examples=args.maxsteps * args.bsize,
            negs_per_query=args.nway-1,
            data_folder=join(args.data_dir, dataset_name),
        )
    elif args.dataset.startswith('mrtydi'):
        dataset_name, language = args.dataset.split('-', 1)
        dataloader = MrTydiColbertLoader(
            lang=language,
            load_train=args.do_train,
            load_test=args.do_test,
            data_folder=join(args.data_dir, dataset_name),
        )
    elif args.dataset == "lleqa":
        dataloader = LLeQAColbertLoader(
            load_test=args.do_test,
            load_dev=True,
            load_train=args.do_train,
            negatives_system='me5',
            max_train_examples=args.maxsteps * args.bsize,
            data_folder=join(args.data_dir, args.dataset),
        )
    data_filepaths = dataloader.load()

    run_kwargs = {
        'rank': 0,
        'amp': True,
        'nranks': torch.cuda.device_count(),
        'root': join(args.output_dir, args.dataset),
        'experiment': 'colbert',
        'name': f'{datetime.now().strftime("%Y-%m-%d_%H.%M")}-{args.model_name.replace("/", "-")}-{args.dataset}',
    }
    model_kwargs = deepcopy(vars(args))
    model_kwargs['checkpoint'] = args.model_name
    for k in {'dataset', 'data_dir', 'output_dir', 'do_train', 'do_test', 'seed'}:
        model_kwargs.pop(k, None)

    if args.do_train:
        with Run().context(RunConfig(**run_kwargs)):
            trainer = CustomTrainer(
                config=ColBERTConfig(**model_kwargs),
                triples=data_filepaths['train_tuples'],
                queries=data_filepaths['train_queries'],
                collection=data_filepaths['collection'],
            )
            trainer.train(checkpoint=args.model_name, seed=args.seed)
            model_kwargs['checkpoint'] = trainer.best_checkpoint_path()

    if args.do_test:
        if 'checkpoints' in model_kwargs['checkpoint']:
            model_path, ckpt_name = model_kwargs['checkpoint'].split('/checkpoints/', 1)
        else:
            model_path, ckpt_name = join(run_kwargs['root'], run_kwargs['experiment'], model_kwargs['checkpoint'].split('/')[-1]), ""
        run_kwargs['index_root'] = join(model_path, 'indexes', ckpt_name)
        with Run().context(RunConfig(**run_kwargs)):
            indexer = CustomIndexer(checkpoint=model_kwargs['checkpoint'], config=ColBERTConfig(**model_kwargs))
            indexer.index(name=f"{args.dataset}.index", collection=data_filepaths['collection'], overwrite='reuse')

        with Run().context(RunConfig(**run_kwargs)):
            split = 'dev' if args.dataset.startswith('mmarco') else 'test'
            queries = Queries(data_filepaths[f"{split}_queries"])
            searcher = CustomSearcher(index=f"{args.dataset}.index", config=ColBERTConfig(**model_kwargs))
            ranking = searcher.search_all(queries, k=1000)
            results_path = ranking.save(f"{args.dataset}-ranking.tsv")
            args.__dict__.update({
                'ranking_filepath': results_path, 
                'qrels_filepath': data_filepaths[f"{split}_qrels"],
                'output_filepath': join(model_path, 'evaluation', ckpt_name, f"results_{args.dataset}.json"),
                'annotate': False, 
                'split': split,
            })
            evaluate(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model settings.
    parser.add_argument("--model_name", type=str, help="The model checkpoint for weights initialization.")
    parser.add_argument("--similarity", type=str, help="Similarity function for scoring query-document representation.", choices=["cosine", "l2"],)
    parser.add_argument("--doc_maxlen", type=int, help="Maximum length at which the passages will be truncated.")
    parser.add_argument("--query_maxlen", type=int, help="Maximum length at which the queries will be truncated.")
    parser.add_argument("--mask_punctuation", action="store_true", default=False, help="Whether to mask punctuation tokens.")
    parser.add_argument("--attend_to_mask_tokens", action="store_true", default=False, help="Whether to attend to mask tokens.")
    # Training settings.
    parser.add_argument("--do_train", action="store_true", default=False, help="Wether to perform training.")
    parser.add_argument("--dim", type=int, help="Dimensionality of the embeddings.")
    parser.add_argument("--bsize", type=int, help="The batch size per GPU/TPU core/CPU for training.")
    parser.add_argument("--accumsteps", type=int, help="The number of accumulation steps before performing a backward/update pass.")
    parser.add_argument("--lr", type=float, help="The initial learning rate for AdamW optimizer.")
    parser.add_argument("--maxsteps", type=int, help="The total number of training steps to perform.")
    parser.add_argument("--warmup", type=int, default=None, help="Number of warmup steps for the learning rate scheduler.")
    parser.add_argument("--nway", type=int, help="Number of passages/documents to compare the query with. Usually, 1 positive passage + k negative passages.")
    parser.add_argument("--use_ib_negatives", action="store_true", default=False, help="Whether to use in-batch negatives during training.")
    parser.add_argument("--distillation_alpha", type=float, help="""Scaling parameter of the target scores when optimizing with KL-divergence loss.
        A higher value increases the differences between the target scores before applying softmax, leading to a more polarized probability distribution.
        A lower value makes the target scores more similar, leading to a softer probability distribution.""")
    parser.add_argument("--ignore_scores", action="store_true", default=False, help="""Whether to ignore scores provided for the n-way tuples. If so, 
        pairwise softmax cross-entropy loss will be applied. Otherwise, KL-divergence loss between the target and log scores will be applied.""")
    parser.add_argument("--seed", type=int, help="Random seed that will be set at the beginning of training.")
    # Index settings (for evaluation).
    parser.add_argument("--do_test", action="store_true", default=False, help="Wether to perform test evaluation after training.")
    parser.add_argument("--nbits", type=int,help="Number of bits for encoding each dimension.")
    parser.add_argument("--kmeans_niters", type=int, help="Number of iterations for k-means clustering. 4 is a good and fast default. Consider larger numbers for small datasets.")
    # Data settings.
    parser.add_argument("--dataset", type=str, help="Dataset to use.", choices=[
        "lleqa", "mmarco-ar", "mmarco-de", "mmarco-en", "mmarco-es", "mmarco-fr", "mmarco-hi", "mmarco-id", 
        "mmarco-it", "mmarco-ja", "mmarco-nl", "mmarco-pt", "mmarco-ru", "mmarco-vi", "mmarco-zh",
    ])
    parser.add_argument("--data_dir", type=str, help="Folder containing the training data.")
    parser.add_argument("--output_dir", type=str, help="Folder to save checkpoints, logs, and evaluation results.")
    # Search settings.
    #parser.add_argument("--ncells", type=int, default=1, help="Number of cells for the IVF index.")
    #parser.add_argument("--centroid_score_threshold", type=float,  default=0.5, help="Threshold for the centroid score.")
    args, _ = parser.parse_known_args()
    main(args)
