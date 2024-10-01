import os
import logging
import argparse
from os.path import join
from datetime import datetime

from torch.utils.data import DataLoader
from sentence_transformers import util

try:
    from src.data.lleqa import LLeQABiencoderLoader
except ModuleNotFoundError:
    import sys, pathlib
    sys.path.append(str(pathlib.Path().resolve()))
from src.data.lleqa import LLeQABiencoderLoader
from src.retrievers.splade import SPLADE, MmarcoReader
from src.utils.loggers import WandbLogger, LoggingHandler
from src.utils.common import set_seed, count_trainable_parameters
from src.utils.sentence_transformers import InformationRetrievalEvaluatorCustom



def main(args):
    set_seed(args.seed)
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, handlers=[LoggingHandler()])

    out_model_name = f"{args.model_name.split('/')[-1]}-{args.dataset}"
    out_path = join(args.output_dir, args.dataset, "splade", f'{datetime.now().strftime("%Y_%m_%d-%H_%M")}-{out_model_name}')

    logging.info("Loading dataset...")
    if args.dataset.startswith('mmarco'):
        args.language = args.dataset.split('-', 1)[-1]
        datareader = MmarcoReader(
            lang=args.language,
            load_dev=args.do_test,
            load_train=args.do_train,
            max_train_examples=args.steps * args.batch_size,
            training_sample_format='triplet', negs_type='hard', negs_mining_systems='all', # Custom
            # training_sample_format='triplet', negs_type='original', # SPLADEv1, SPLADEv2
            # training_sample_format='tuple_with_scores', negs_type='hard', negs_mining_systems='bm25', # SPLADEplus
            # training_sample_format='tuple_with_scores', negs_type='hard', negs_mining_systems='all', # SPLADEplusEnsemble, SPLADEeff
            # training_sample_format='tuple_with_scores', negs_type='hard', negs_mining_systems='all', negs_per_query=2, # SPLADEv3
        )
    elif args.dataset == 'lleqa':
        datareader = LLeQABiencoderLoader(load_train=args.do_train, load_dev=True, load_test=args.do_test)
        datareader.negs_per_query = 1
    data = datareader.load()

    logging.info("Loading neural sparse model...")
    model = SPLADE(
        model_name_or_path=args.model_name, 
        pooling=args.pooling,
        similarity=args.sim,
        max_query_length=args.query_maxlen, 
        max_doc_length=args.doc_maxlen,
        #augment_query_to_maxlen=True,
        #augment_doc_to_maxlen=False,
        #query_prefix="<s>NOTUSED",
        #doc_prefix="</s>NOTUSED", #print(self.model.tokenizer.additional_special_tokens)
        device=args.device,
    )
    args.model_params = count_trainable_parameters(model.model)

    os.makedirs(join(args.output_dir, 'logs'), exist_ok=True)
    logger = WandbLogger(project_name=args.dataset, run_name=f"splade-{out_model_name}", run_config=args, log_dir=join(args.output_dir, 'logs'))

    if args.do_train:
        logging.info("Training...")
        train_dataloader = DataLoader(
            data['train'], shuffle=True, batch_size=args.batch_size, 
            collate_fn=lambda b: model.collate(b, datareader.negs_per_query),
        )
        model.fit(
            dataloader=train_dataloader,
            rank_loss_config={'name': 'InfoNCELoss', 'temperature': 0.05, 'use_ib_negs': True},
            steps=args.steps,
            fp16_amp=args.use_fp16, 
            bf16_amp=args.use_bf16,
            scheduler=args.scheduler, 
            warmup_ratio=args.warmup_ratio,
            optimizer_name=args.optimizer,
            learning_rate=args.lr,
            weight_decay=args.wd,
            log_callback=logger.log_training,
            log_every_n_steps=args.log_steps,
            output_path=out_path,
            ckpt_path=out_path if args.save_during_training else None, 
            ckpt_save_steps=int(len(train_dataloader) / 10), 
            ckpt_save_limit=3,
        )
        model.save(f"{out_path}/final")

    if args.do_test:
        logging.info("Testing...")
        split = 'dev' if args.dataset.startswith('mmarco') else 'test'
        test_evaluator = InformationRetrievalEvaluatorCustom(
            name=split, 
            corpus=data['corpus'], queries=data[split]['queries'], relevant_docs=data[split]['labels'],
            precision_recall_at_k=[5, 10, 20, 50, 100, 200, 500, 1000], 
            map_at_k=[10, 100], mrr_at_k=[10, 100], ndcg_at_k=[10, 100], accuracy_at_k=[1],
            batch_size=args.batch_size, corpus_chunk_size=50000, score_functions={model.similarity: getattr(util, model.similarity)},
            log_callback=logger.log_eval, show_progress_bar=True,
        )
        model.evaluate(evaluator=test_evaluator, output_path=out_path, epoch=0, steps=args.steps if args.do_train else 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset.
    parser.add_argument("--dataset", type=str, help="Dataset to use.", choices=[
        "lleqa", "mmarco-ar", "mmarco-de", "mmarco-en", "mmarco-es", "mmarco-fr", "mmarco-hi", "mmarco-id", 
        "mmarco-it", "mmarco-ja", "mmarco-nl", "mmarco-pt", "mmarco-ru", "mmarco-vi", "mmarco-zh",
    ])
    # Model.
    parser.add_argument("--model_name", type=str, help="The model checkpoint for weights initialization.")
    parser.add_argument("--query_maxlen", type=int, help="Maximum length at which the queries will be truncated.")
    parser.add_argument("--doc_maxlen", type=int, help="Maximum length at which the passages will be truncated.")
    parser.add_argument("--pooling", type=str, help="Type of pooling to perform to get a passage representation.", choices=["sum", "max", "cls"])
    parser.add_argument("--sim", type=str, help="Similarity function for scoring query-document representation.", choices=["cos_sim", "dot_score"])
    # Training.
    parser.add_argument("--do_train", action="store_true", default=False, help="Wether to perform training.")
    parser.add_argument("--steps", type=int, help="Number of training steps.")
    parser.add_argument("--batch_size", type=int, help="The batch size per GPU/TPU core/CPU for training.")
    parser.add_argument("--optimizer", type=str, help="Type of optimizer to use for training.", choices=["AdamW", "Adafactor", "Shampoo"])
    parser.add_argument("--lr", type=float, help="The initial learning rate for AdamW optimizer.")
    parser.add_argument("--wd", type=float, help="The weight decay to apply (if not zero) to all layers in AdamW optimizer.")
    parser.add_argument("--scheduler", type=str, help="Type of learning rate scheduler to use for training.", choices=[
        "constant", "constant_with_warmup", "linear", "polynomial", "inverse_sqrt", "cosine", "cosine_with_restarts", "cosine_with_min_lr", "reduce_lr_on_plateau",
    ])
    parser.add_argument("--warmup_ratio", type=float, help="Ratio of total training steps used for a linear warmup from 0 to 'lr'.")
    parser.add_argument("--use_fp16", action="store_true", default=False, help="Whether to use mixed precision during training.")
    parser.add_argument("--use_bf16", action="store_true", default=False, help=".")
    parser.add_argument("--seed", type=int, help="Random seed that will be set at the beginning of training.")
    parser.add_argument("--save_during_training", action="store_true", default=False, help="Wether to save model checkpoints during training.")
    parser.add_argument("--log_steps", type=int, help="Log every k training steps.")
    parser.add_argument("--output_dir", type=str, help="Folder to save checkpoints, logs, and evaluation results.")
    # Testing.
    parser.add_argument("--do_test", action="store_true", default=False, help="Wether to perform test evaluation.")
    parser.add_argument("--device", type=str, help="The device on which to perform the run.", choices=["cuda", "cpu"])
    args, _ = parser.parse_known_args()
    main(args)
