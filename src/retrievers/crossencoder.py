import os
import logging
import argparse
from os.path import join
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from torch.utils.data import DataLoader
from transformers import logging as hf_logger
hf_logger.set_verbosity_error()

try:
    from src.data.lleqa import LLeQACrossencoderLoader
except ModuleNotFoundError:
    import sys, pathlib
    sys.path.append(str(pathlib.Path().resolve()))
from src.data.lleqa import LLeQACrossencoderLoader
from src.data.mmarco import MmarcoCrossencoderLoader
from src.utils.loggers import WandbLogger, LoggingHandler
from src.utils.common import set_seed, count_trainable_parameters
from src.utils.sentence_transformers import CrossEncoderCustom, CERerankingEvaluatorCustom


def main(args):
    set_seed(args.seed)
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, handlers=[LoggingHandler()])

    out_model_name = f"{args.model_name.split('/')[-1]}-{args.dataset}"
    out_path = join(args.output_dir, args.dataset, "crossencoder", f'{datetime.now().strftime("%Y_%m_%d-%H_%M")}-{out_model_name}')

    logging.info("Loading dataset...")
    if args.dataset.startswith('mmarco'):
        args.language = args.dataset.split('-', 1)[-1]
        dataloader = MmarcoCrossencoderLoader(
            lang=args.language,
            load_train=args.do_train,
            max_train_examples=args.max_steps * args.batch_size,
            load_dev=(args.eval_during_training or args.do_test),
        )
    elif args.dataset == 'lleqa':
        dataloader = LLeQACrossencoderLoader(
            load_train=args.do_train,
            max_train_examples=args.max_steps * args.batch_size,
            load_dev=True,
            load_test=args.do_test,
        )
    data = dataloader.load()

    logging.info("Loading cross-encoder model...")
    model = CrossEncoderCustom(
        model_name=args.model_name, 
        num_labels=1, 
        max_length=args.max_seq_length, 
        automodel_args={'token': os.getenv("HF")}, 
        tokenizer_args={'model_max_length': args.max_seq_length},
        device=args.device,
    )
    args.model_params = count_trainable_parameters(model.model)

    os.makedirs(join(args.output_dir, 'logs'), exist_ok=True)
    logger = WandbLogger(project_name=args.dataset, run_name=f"crossencoder-{out_model_name}", run_config=args, log_dir=join(args.output_dir, 'logs'))

    if args.do_train:
        logging.info("Training...")
        train_dataloader = DataLoader(data['train'], shuffle=False, batch_size=args.batch_size)
        saving_steps = max(1e3, int(args.max_steps/10))

        dev_evaluator, eval_steps = None, 0
        if args.eval_during_training:
            dev_evaluator = CERerankingEvaluatorCustom(name='dev', samples=data['dev'], k=[5, 10, 50, 100, 500, 1000], log_callback=logger.log_eval)
            eval_steps = saving_steps

        model.fit(
            train_dataloader=train_dataloader, epochs=1, 
            fp16_amp=args.use_fp16, bf16_amp=args.use_bf16,
            scheduler=args.scheduler, warmup_steps=int(args.warmup_ratio * args.max_steps),
            optimizer_class=getattr(sys.modules[__name__], args.optimizer), weight_decay=args.wd, 
            optimizer_params={'lr': args.lr, **(
                {'clip_threshold': 1.0, 'scale_parameter': False, 'relative_step': False, 'warmup_init': False} if args.optimizer == "Adafactor" 
                else {'no_deprecation_warning': True} if args.optimizer == "AdamW" else {}
            )},
            evaluator=dev_evaluator, evaluation_steps=eval_steps, output_path=out_path,
            log_every_n_steps=args.log_steps, log_callback=logger.log_training, show_progress_bar=True,
            checkpoint_path=out_path if args.save_during_training else None, checkpoint_save_steps=saving_steps, checkpoint_save_total_limit=3,
        )
        model.save(f"{out_path}/final")

    if args.do_test:
        logging.info("Testing...")
        split = 'dev' if args.dataset.startswith('mmarco') else 'test'
        test_evaluator = CERerankingEvaluatorCustom(name=split, samples=data[split],  k=[5, 10, 50, 100, 500, 1000], log_callback=logger.log_eval)
        model.evaluate(evaluator=test_evaluator, output_path=out_path, epoch=0, steps=args.max_steps if args.do_train else 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset to use.", choices=[
        "lleqa", "mmarco-ar", "mmarco-de", "mmarco-en", "mmarco-es", "mmarco-fr", "mmarco-hi", "mmarco-id", 
        "mmarco-it", "mmarco-ja", "mmarco-nl", "mmarco-pt", "mmarco-ru", "mmarco-vi", "mmarco-zh",
    ])
    parser.add_argument("--model_name", type=str, help="The Hugging Face model name for weights initialization.")
    parser.add_argument("--max_seq_length", type=int, help="Maximum length at which the query+passage sequences will be truncated.")
    parser.add_argument("--do_train", action="store_true", default=False, help="Wether to perform training.")
    parser.add_argument("--max_steps", type=int, help="Total number of training steps to perform.")
    parser.add_argument("--batch_size", type=int, help="The batch size per GPU/TPU core/CPU for training.")
    parser.add_argument("--optimizer", type=str, help="Type of optimizer to use for training.", choices=["AdamW", "Adafactor", "Shampoo"])
    parser.add_argument("--lr", type=float, help="The initial learning rate for the optimizer.")
    parser.add_argument("--wd", type=float, help="The weight decay to apply (if not zero) to all layers in AdamW optimizer.")
    parser.add_argument("--scheduler", type=str, help="Type of learning rate scheduler to use for training.", choices=[
        "constantlr", "warmupconstant", "warmuplinear", "warmupcosine", "warmupcosinewithhardrestarts"
    ])
    parser.add_argument("--warmup_ratio", type=float, help="Ratio of total training steps used for a linear warmup from 0 to 'lr'.")
    parser.add_argument("--use_fp16", action="store_true", default=False, help="Whether to use mixed precision during training.")
    parser.add_argument("--use_bf16", action="store_true", default=False, help="Whether to use bfloat16 mixed precision during training.")
    parser.add_argument("--seed", type=int, help="Random seed that will be set at the beginning of training.")
    parser.add_argument("--save_during_training", action="store_true", default=False, help="Whether to save model checkpoints during training.")
    parser.add_argument("--eval_during_training", action="store_true", default=False, help="Whether to perform dev evaluation during training.")
    parser.add_argument("--log_steps", type=int, help="Log every k training steps.")
    parser.add_argument("--output_dir", type=str, help="Folder to save checkpoints, logs, and evaluation results.")
    parser.add_argument("--do_test", action="store_true", default=False, help="Whether to perform test evaluation.")
    parser.add_argument("--device", type=str, help="The device on which to perform the run.", choices=["cuda", "cpu"])
    args, _ = parser.parse_known_args()
    main(args)
