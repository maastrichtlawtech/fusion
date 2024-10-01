import os
import logging
import argparse
from os.path import join
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import wandb
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import AdamW, Adafactor
from transformers import logging as hf_logger
hf_logger.set_verbosity_error()

from sentence_transformers import util, LoggingHandler
from sentence_transformers.losses import MultipleNegativesRankingLoss

try:
    from src.utils.optim import Shampoo
except ModuleNotFoundError:
    import sys, pathlib
    sys.path.append(str(pathlib.Path().resolve()))
from src.utils.optim import Shampoo
from src.utils.loggers import WandbLogger
from src.data.lleqa import LLeQABiencoderLoader
from src.data.mmarco import MmarcoBiencoderLoader
from src.utils.sentence_transformers import InformationRetrievalEvaluatorCustom
from src.utils.common import set_seed, load_sbert_model, count_trainable_parameters, set_xmod_language, prepare_xmod_for_finetuning


def main(args):
    set_seed(args.seed)
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, handlers=[LoggingHandler()])

    out_model_name = f"{args.model_name.split('/')[-1]}-{args.dataset}"
    out_path = join(args.output_dir, args.dataset, "biencoder", f'{datetime.now().strftime("%Y_%m_%d-%H_%M")}-{out_model_name}')

    logging.info("Loading dataset...")
    if args.dataset.startswith('mmarco'):
        args.language = args.dataset.split('-', 1)[-1]
        datareader = MmarcoBiencoderLoader(lang=args.language, load_train=args.do_train, load_dev=args.do_test)
    elif args.dataset == 'lleqa':
        args.language = 'fr'
        datareader = LLeQABiencoderLoader(load_train=args.do_train, load_dev=True, load_test=args.do_test, return_st_format=True)
    data = datareader.load()
    
    logging.info("Loading biencoder model...")
    model = load_sbert_model(model_name=args.model_name, max_seq_length=args.max_seq_length, pooling=args.pooling, device=args.device)
    args.model_params = count_trainable_parameters(model)
    
    os.makedirs(join(args.output_dir, 'logs'), exist_ok=True)
    logger = WandbLogger(project_name=args.dataset, run_name=f"biencoder-{out_model_name}", run_config=args, log_dir=join(args.output_dir, 'logs'))

    if args.do_train:
        logging.info("Training...")
        loss = MultipleNegativesRankingLoss(model=model, similarity_fct=getattr(util, args.sim))

        train_dataloader = DataLoader(data['train'], shuffle=True, batch_size=args.batch_size)
        total_steps = args.epochs * len(train_dataloader)

        dev_evaluator, eval_steps = None, 0
        if args.eval_during_training:
            dev_evaluator = InformationRetrievalEvaluatorCustom(
                name='dev', corpus=data['corpus'], queries=data['dev']['queries'], relevant_docs=data['dev']['labels'], 
                precision_recall_at_k=[5, 10, 20, 50, 100, 200, 500, 1000], 
                map_at_k=[10, 100], mrr_at_k=[10, 100], ndcg_at_k=[10, 100], accuracy_at_k=[],
                batch_size=args.batch_size, corpus_chunk_size=50000, score_functions={args.sim: getattr(util, args.sim)},
                log_callback=logger.log_eval, show_progress_bar=True,
            )
            eval_steps = len(train_dataloader)

        model.fit(
            train_objectives=[(train_dataloader, loss)],
            epochs=args.epochs,
            fp16_amp=args.use_fp16, bf16_amp=args.use_bf16,
            scheduler=args.scheduler, warmup_steps=int(args.warmup_ratio * total_steps),
            optimizer_class=getattr(sys.modules[__name__], args.optimizer), weight_decay=args.wd, 
            optimizer_params={'lr': args.lr, **(
                {'clip_threshold': 1.0, 'scale_parameter': False, 'relative_step': False, 'warmup_init': False} if args.optimizer == "Adafactor" 
                else {'no_deprecation_warning': True} if args.optimizer == "AdamW" else {}
            )},
            evaluator=dev_evaluator, evaluation_steps=eval_steps, output_path=out_path,
            log_every_n_steps=args.log_steps, log_callback=logger.log_training, show_progress_bar=True,
            checkpoint_path=out_path if args.save_during_training else None, checkpoint_save_steps=len(train_dataloader), checkpoint_save_total_limit=3,
        )
        model.save(f"{out_path}/final")

    if args.do_test:
        logging.info("Testing...")
        split = 'dev' if args.dataset.startswith('mmarco') else 'test'
        test_evaluator = InformationRetrievalEvaluatorCustom(
            name=split, corpus=data['corpus'], queries=data[split]['queries'], relevant_docs=data[split]['labels'],
            precision_recall_at_k=[5, 10, 20, 50, 100, 200, 500, 1000], 
            map_at_k=[10, 100], mrr_at_k=[10, 100], ndcg_at_k=[10, 100], accuracy_at_k=[1],
            batch_size=args.batch_size, corpus_chunk_size=50000, score_functions={args.sim: getattr(util, args.sim)},
            log_callback=logger.log_eval, show_progress_bar=True,
        )
        model.evaluate(evaluator=test_evaluator, output_path=out_path, epoch=args.epochs if args.do_train else 0, steps=total_steps if args.do_train else 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset.
    parser.add_argument("--dataset", type=str, help="Dataset to use.", choices=[
        "lleqa", "mmarco-ar", "mmarco-de", "mmarco-en", "mmarco-es", "mmarco-fr", "mmarco-hi", "mmarco-id", 
        "mmarco-it", "mmarco-ja", "mmarco-nl", "mmarco-pt", "mmarco-ru", "mmarco-vi", "mmarco-zh",
    ])
    # Model.
    parser.add_argument("--model_name", type=str, help="The model checkpoint for weights initialization.")
    parser.add_argument("--max_seq_length", type=int, help="Maximum length at which the passages will be truncated.")
    parser.add_argument("--pooling", type=str, help="Type of pooling to perform to get a passage representation.", choices=["mean", "max", "cls"])
    parser.add_argument("--sim", type=str, help="Similarity function for scoring query-document representation.", choices=["cos_sim", "dot_score"])
    # Training.
    parser.add_argument("--do_train", action="store_true", default=False, help="Wether to perform training.")
    parser.add_argument("--epochs", type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", type=int, help="The batch size per GPU/TPU core/CPU for training.")
    parser.add_argument("--optimizer", type=str, help="Type of optimizer to use for training.", choices=["AdamW", "Adafactor", "Shampoo"])
    parser.add_argument("--lr", type=float, help="The initial learning rate for AdamW optimizer.")
    parser.add_argument("--wd", type=float, help="The weight decay to apply (if not zero) to all layers in AdamW optimizer.")
    parser.add_argument("--scheduler", type=str, help="Type of learning rate scheduler to use for training.", choices=[
        "constantlr", "warmupconstant", "warmuplinear", "warmupcosine", "warmupcosinewithhardrestarts"
    ])
    parser.add_argument("--warmup_ratio", type=float, help="Ratio of total training steps used for a linear warmup from 0 to 'lr'.")
    parser.add_argument("--use_fp16", action="store_true", default=False, help="Whether to use mixed precision during training.")
    parser.add_argument("--use_bf16", action="store_true", default=False, help="Whether to use bfloat16 mixed precision during training.")
    parser.add_argument("--seed", type=int, help="Random seed that will be set at the beginning of training.")
    parser.add_argument("--save_during_training", action="store_true", default=False, help="Wether to save model checkpoints during training.")
    parser.add_argument("--eval_during_training", action="store_true", default=False, help="Wether to perform dev evaluation during training.")
    parser.add_argument("--log_steps", type=int, help="Log every k training steps.")
    parser.add_argument("--output_dir", type=str, help="Folder to save checkpoints, logs, and evaluation results.")
    # Testing.
    parser.add_argument("--do_test", action="store_true", default=False, help="Wether to perform test evaluation.")
    parser.add_argument("--device", type=str, help="The device on which to perform the run.", choices=["cuda", "cpu"])
    args, _ = parser.parse_known_args()
    main(args)
