
"""
Custom SentenceTransformer class with some extra features.
"""
import os
import csv
import json
import time
import shutil
import inspect
import logging
from contextlib import nullcontext
from tqdm.autonotebook import tqdm, trange
from typing import Dict, Tuple, Iterable, Callable, Type, List, Set, Union, Optional

import heapq
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer
from sentence_transformers.model_card_templates import ModelCardTemplate
from sentence_transformers.util import fullname, batch_to_device, cos_sim, dot_score, import_from_string
from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction, MSEEvaluator, EmbeddingSimilarityEvaluator, InformationRetrievalEvaluator
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, T5Config, MT5Config

from .t5 import T5EncoderForSequenceClassification, MT5EncoderForSequenceClassification, clean_t5_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------#
#                                               BI-ENCODER
#-----------------------------------------------------------------------------------------------------------------#
class SentenceTransformerCustom(SentenceTransformer):
    def fit(
        self,
        train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        steps_per_epoch = None,
        scheduler: str = 'WarmupLinear',
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0,
        # New
        log_callback: Callable[[int, int, int, float, float], None] = None,
        log_every_n_steps: int = 0,
        fp16_amp: bool = False,
        bf16_amp: bool = False,
    ):
        """
        Custom fit() function with some extra features.
        - a generic logging framework parameter: https://github.com/UKPLab/sentence-transformers/pull/1606#issuecomment-1383608304
        - support for bf16 training: https://github.com/UKPLab/sentence-transformers/pull/2285
        """
        assert not (fp16_amp and bf16_amp), f"Both fp16 and bf16 training have been set to True. Please choose one only."
        info_loss_functions =  []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps({"evaluator": fullname(evaluator), "epochs": epochs, "steps_per_epoch": steps_per_epoch, "scheduler": scheduler, "warmup_steps": warmup_steps, "optimizer_class": str(optimizer_class),  "optimizer_params": optimizer_params, "weight_decay": weight_decay, "evaluation_steps": evaluation_steps, "max_grad_norm": max_grad_norm }, indent=4, sort_keys=True)
        self._model_card_text = None
        self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}", info_loss_functions).replace("{FIT_PARAMETERS}", info_fit_parameters)

        if fp16_amp or bf16_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self._target_device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data
                    labels = labels.to(self._target_device)
                    features = list(map(lambda batch: batch_to_device(batch, self._target_device), features))

                    fwd_pass_context = autocast(dtype=torch.bfloat16) if bf16_amp else (autocast(dtype=torch.float16) if fp16_amp else nullcontext())
                    with fwd_pass_context:
                        loss_value = loss_model(features, labels)

                    if fp16_amp:
                        # Scaler isn't necessary for bfloat16
                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                    if log_callback is not None and log_every_n_steps > 0 and training_steps % log_every_n_steps == (log_every_n_steps - 1):
                        try:
                            log_callback(train_idx, epoch, global_step, scheduler.get_last_lr()[0], loss_value.item())
                        except Exception as e:
                            logger.warning(f"Logging error encountered: {e}. Ignoring..")

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, global_step, callback)

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                    self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

        if evaluator is None and output_path is not None:
            self.save(output_path)

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


    def evaluate(self, evaluator: SentenceEvaluator, output_path: str = None, epoch: int = -1, steps: int = -1):
        """Custom .evaluate() function with two extra 'epoch' and 'steps' parameters.
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        return evaluator(self, output_path=output_path, epoch=epoch, steps=steps)


class InformationRetrievalEvaluatorCustom(InformationRetrievalEvaluator):
    def __init__(self,
        queries: Dict[str, str],  #qid => query
        corpus: Dict[str, str],  #cid => doc
        relevant_docs: Dict[str, Set[str]],  #qid => Set[cid]
        corpus_chunk_size: int = 50000,
        mrr_at_k: List[int] = [10],
        ndcg_at_k: List[int] = [10],
        accuracy_at_k: List[int] = [1, 3, 5, 10],
        precision_recall_at_k: List[int] = [1, 3, 5, 10],
        map_at_k: List[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = '',
        write_csv: bool = True,
        score_functions: List[Callable[[Tensor, Tensor], Tensor] ] = {'cos_sim': cos_sim, 'dot_score': dot_score},
        main_score_function: str = None,
        # New
        log_callback: Callable[[int, int, str, str, float], None] = None,
    ):
        """Custom init function that adds an extra generic logging framework 'log_callback' and an 'output_value' 
        parameter indicating if the model outputs one sentence embedding or all token embeddings (for ColBERT).
        """
        super().__init__(
            queries, corpus, relevant_docs, corpus_chunk_size, 
            mrr_at_k, ndcg_at_k, accuracy_at_k, precision_recall_at_k, map_at_k,
            show_progress_bar, batch_size, name, write_csv, score_functions, main_score_function,
        )
        self.log_callback = log_callback
    
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs) -> float:
        """Custom call function that logs scores to the provided logging framework and has two extra parameters:
        'epoch' and 'steps'.
        """
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"
        logger.info("Information Retrieval Evaluation on " + self.name + " dataset" + out_txt)

        scores = self.compute_metrices(model, *args, **kwargs)

        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")
            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    val = scores[name]['accuracy@k'][k]
                    output_data.append(val)
                    if self.log_callback:
                        self.log_callback(epoch, steps, f'{self.name}/accuracy@{k}', val)

                for k in self.precision_recall_at_k:
                    p_val = scores[name]['precision@k'][k]
                    r_val = scores[name]['recall@k'][k]
                    output_data.append(p_val)
                    output_data.append(r_val)
                    if self.log_callback:
                        self.log_callback(epoch, steps, f'{self.name}/precision@{k}', p_val)
                        self.log_callback(epoch, steps, f'{self.name}/recall@{k}', r_val)

                for k in self.mrr_at_k:
                    val = scores[name]['mrr@k'][k]
                    output_data.append(val)
                    if self.log_callback:
                        self.log_callback(epoch, steps, f'{self.name}/mrr@{k}', val)

                for k in self.ndcg_at_k:
                    val = scores[name]['ndcg@k'][k]
                    output_data.append(val)
                    if self.log_callback:
                        self.log_callback(epoch, steps, f'{self.name}/ndcg@{k}', val)

                for k in self.map_at_k:
                    val = scores[name]['map@k'][k]
                    output_data.append(val)
                    if self.log_callback:
                        self.log_callback(epoch, steps, f'{self.name}/map@{k}', val)

                if self.log_callback:
                    self.log_callback(epoch, steps, f'{self.name}/r-precision', scores[name]['r-precision'])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if self.main_score_function is None:
            return max([scores[name]['map@k'][max(self.map_at_k)] for name in self.score_function_names])
        else:
            return scores[self.main_score_function]['map@k'][max(self.map_at_k)]


    def compute_metrices(self, model, corpus_model = None, corpus_embeddings: Tensor = None) -> Dict[str, float]:
        if corpus_model is None:
            corpus_model = model
        max_k = max(max(self.mrr_at_k), max(self.ndcg_at_k), max(self.accuracy_at_k), max(self.precision_recall_at_k), max(self.map_at_k))

        # Compute embedding for the queries
        kwargs = {'sentences': self.queries, 'show_progress_bar': self.show_progress_bar, 'batch_size': self.batch_size, 'convert_to_tensor': True}
        if 'query_mode' in inspect.signature(model.encode).parameters:
            kwargs.update({'query_mode': True})
        t0 = time.perf_counter()
        query_embeddings = model.encode(**kwargs)
        t1 = time.perf_counter()
        encoding_latency = ((t1 - t0) / len(self.queries)) * 1000
        
        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        #Iterate over chunks of the corpus
        scoring_latencies = []
        for corpus_start_idx in trange(0, len(self.corpus), int(self.corpus_chunk_size), desc='Corpus Chunks', disable=not self.show_progress_bar):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            #Encode chunk of corpus
            if corpus_embeddings is None:
                kwargs = {'sentences': self.corpus[corpus_start_idx:corpus_end_idx], 'show_progress_bar': self.show_progress_bar, 'batch_size': 128, 'convert_to_tensor': True}
                if 'query_mode' in inspect.signature(model.encode).parameters:
                    kwargs.update({'query_mode': False})
                sub_corpus_embeddings = model.encode(**kwargs)
            else:
                sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            #Compute similarity scores
            for name, score_function in self.score_functions.items():
                pair_scores_top_k_values, pair_scores_top_k_idx = [], []

                t0 = time.perf_counter()
                for emb in query_embeddings:
                    pair_scores = score_function(emb.unsqueeze(0), sub_corpus_embeddings)
                    topk_values, topk_idx = torch.topk(pair_scores, min(max_k, pair_scores.size(1)), dim=1, largest=True, sorted=False)
                    pair_scores_top_k_values.append(topk_values.cpu().tolist()[0])
                    pair_scores_top_k_idx.append(topk_idx.cpu().tolist()[0])
                t1 = time.perf_counter()

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]):
                        corpus_id = self.corpus_ids[corpus_start_idx+sub_corpus_id]
                        if len(queries_result_list[name][query_itr]) < max_k:
                            heapq.heappush(queries_result_list[name][query_itr], (score, corpus_id))  # heaqp tracks the quantity of the first element in the tuple
                        else:
                            heapq.heappushpop(queries_result_list[name][query_itr], (score, corpus_id))
            
            scoring_latencies.append(t1-t0)
        scoring_latency = (np.sum(scoring_latencies) / len(self.queries)) * 1000

        t0 = time.perf_counter()
        for name in queries_result_list:
            for query_itr in range(len(queries_result_list[name])):
                for doc_itr in range(len(queries_result_list[name][query_itr])):
                    score, corpus_id = queries_result_list[name][query_itr][doc_itr]
                    queries_result_list[name][query_itr][doc_itr] = {'corpus_id': corpus_id, 'score': score}
        t1 = time.perf_counter()
        formatting_latency = ((t1 - t0) / len(self.queries)) * 1000

        latency = encoding_latency + scoring_latency + formatting_latency
        if self.log_callback:
            self.log_callback(0, 0, 'latency (ms/q)', latency)
        logger.info(f"Avg. latency (ms/query): {latency:.2f} (Encoding: {encoding_latency:.2f}; Scoring: {scoring_latency:.2f}; Formatting: {formatting_latency:.2f})")

        logger.info("Queries: {}".format(len(self.queries)))
        logger.info("Corpus: {}\n".format(len(self.corpus)))

        #Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        #Output
        for name in self.score_function_names:
            logger.info("Score-Function: {}".format(name))
            self.output_scores(scores[name])
        return scores

    def compute_metrics(self, queries_result_list: List[object]):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precision = {k: [] for k in self.precision_recall_at_k}
        recall = {k: [] for k in self.precision_recall_at_k}
        mrr = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        avgP = {k: [] for k in self.map_at_k}
        rp = []

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x["score"], reverse=True)
            query_relevant_docs = self.relevant_docs[query_id]
            num_relevant_docs = len(query_relevant_docs)

            # Accuracy@k - We count the result correct, if at least one relevant doc is across the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                precision[k_val].append(num_correct / k_val)
                recall[k_val].append(num_correct / num_relevant_docs)

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        mrr[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [1 if top_hit["corpus_id"] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]]
                true_relevances = [1] * num_relevant_docs
                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(true_relevances, k_val)
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)
                avg_precision = sum_precisions / min(k_val, num_relevant_docs)
                avgP[k_val].append(avg_precision)

            # R-Precision
            num_correct = 0
            for hit in top_hits[0:num_relevant_docs]:
                if hit["corpus_id"] in query_relevant_docs:
                    num_correct += 1
            rp.append(num_correct / num_relevant_docs)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)
        for k in precision:
            precision[k] = np.mean(precision[k])
        for k in recall:
            recall[k] = np.mean(recall[k])
        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])
        for k in mrr:
            mrr[k] /= len(self.queries)
        for k in avgP:
            avgP[k] = np.mean(avgP[k])
        rp_avg = np.mean(rp)

        return {
            "accuracy@k": num_hits_at_k,
            "precision@k": precision,
            "recall@k": recall,
            "ndcg@k": ndcg,
            "mrr@k": mrr,
            "map@k": avgP,
            "r-precision": rp_avg,
        }

    def output_scores(self, scores):
        for metric, val in scores.items():
            if isinstance(val, dict):
                for k, score in val.items():
                    logger.info(f"{metric.split('@k')[0].capitalize()}@{k}: {score:.3f}")
            else:
                logger.info(f"{metric.capitalize()}: {val:.3f}")

#-----------------------------------------------------------------------------------------------------------------#
#                                                   CROSS-ENCODER
#-----------------------------------------------------------------------------------------------------------------#
class CrossEncoderCustom(CrossEncoder):
    def __init__(
        self,
        model_name: str,
        num_labels: int = None,
        max_length: int = None,
        device: str = None,
        tokenizer_args: Dict = {},
        automodel_args: Dict = {},
        revision: Optional[str] = None,
        default_activation_function=None,
        classifier_dropout: float = None,
    ):
        self.config = AutoConfig.from_pretrained(model_name)
        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in self.config.architectures])

        if classifier_dropout is not None:
            self.config.classifier_dropout = classifier_dropout

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels

        if isinstance(self.config, T5Config):
            clean_t5_config(self.config, model_type='t5')
            self.model = T5EncoderForSequenceClassification.from_pretrained(
                model_name, config=self.config, revision=revision, **automodel_args
            )
        elif isinstance(self.config, MT5Config):
            clean_t5_config(self.config, model_type='t5')
            self.model = MT5EncoderForSequenceClassification.from_pretrained(
                model_name, config=self.config, revision=revision, **automodel_args
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, config=self.config, revision=revision, **automodel_args
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

        if default_activation_function is not None:
            self.default_activation_function = default_activation_function
            try:
                self.config.sbert_ce_default_activation_function = fullname(self.default_activation_function)
            except Exception as e:
                logger.warning("Was not able to update config about the default_activation_function: {}".format(str(e)) )
        elif hasattr(self.config, 'sbert_ce_default_activation_function') and self.config.sbert_ce_default_activation_function is not None:
            self.default_activation_function = import_from_string(self.config.sbert_ce_default_activation_function)()
        else:
            self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()

    
    def fit(self,
        train_dataloader: DataLoader,
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        loss_fct = None,
        activation_fct = nn.Identity(),
        scheduler: str = 'WarmupLinear',
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Dict[str, object] = {'lr': 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        # New
        checkpoint_path: str = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0,
        log_callback: Callable[[int, int, int, float, float], None] = None,
        log_every_n_steps: int = 0,
        fp16_amp: bool = False,
        bf16_amp: bool = False,
    ):
        """
        Custom fit() function with some extra features.
        - checkpoint saving
        - a generic logging framework parameter: https://github.com/UKPLab/sentence-transformers/pull/1606#issuecomment-1383608304
        - support for bf16 training: https://github.com/UKPLab/sentence-transformers/pull/2285
        """
        train_dataloader.collate_fn = self.smart_batching_collate

        if fp16_amp or bf16_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()

        global_step = 0
        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                fwd_pass_context = autocast(dtype=torch.bfloat16) if bf16_amp else (autocast(dtype=torch.float16) if fp16_amp else nullcontext())
                with fwd_pass_context:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)

                if fp16_amp:
                    # Scaler isn't necessary for bfloat16
                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer) 
                    scaler.update()
                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                if log_callback is not None and log_every_n_steps > 0 and training_steps % log_every_n_steps == (log_every_n_steps - 1): 
                    try:
                        log_callback(0, epoch, global_step, scheduler.get_last_lr()[0], loss_value.item())
                    except Exception as e:
                        logger.warning(f"Logging error encountered: {e}. Ignoring..")

                training_steps += 1
                global_step += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, global_step, callback)

                    self.model.zero_grad()
                    self.model.train()

                if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                    self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

        if evaluator is None and output_path is not None:   #No evaluator, but output path: save final model version
            self.save(output_path)

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


    def evaluate(self, evaluator: SentenceEvaluator, output_path: str = None, epoch: int = -1, steps: int = -1):
        """Custom .evaluate() function with two extra 'epoch' and 'steps' parameters.
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        return evaluator(self, output_path=output_path, epoch=epoch, steps=steps)


    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        eval_path = output_path
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            eval_path = os.path.join(output_path, "eval")
            os.makedirs(eval_path, exist_ok=True)

        if evaluator is not None:
            score = evaluator(self, output_path=eval_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def _save_checkpoint(self, checkpoint_path, checkpoint_save_total_limit, step):
        # Store new checkpoint
        self.save(os.path.join(checkpoint_path, str(step)))

        # Delete old checkpoints
        if checkpoint_save_total_limit is not None and checkpoint_save_total_limit > 0:
            old_checkpoints = []
            for subdir in os.listdir(checkpoint_path):
                if subdir.isdigit():
                    old_checkpoints.append({'step': int(subdir), 'path': os.path.join(checkpoint_path, subdir)})

            if len(old_checkpoints) > checkpoint_save_total_limit:
                old_checkpoints = sorted(old_checkpoints, key=lambda x: x['step'])
                shutil.rmtree(old_checkpoints[0]['path'])


class CERerankingEvaluatorCustom(CERerankingEvaluator):
    def __init__(self, 
        samples, 
        name: str = '', 
        write_csv: bool = True,
        k: List[int] = [10], 
        log_callback: Callable[[int, int, str, str, float], None] = None,
    ):
        super().__init__(samples=samples, name=name, write_csv=write_csv)
        self.log_callback = log_callback
        self.k = k
        self.csv_headers = ["epoch", "steps", "cutoff", "mrr", "recall", "r-precision"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1):
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"
        logger.info("Information Retrieval Evaluation on " + self.name + " dataset" + out_txt)

        all_r_precisions = []
        all_mrr_scores = {cutoff: [] for cutoff in self.k}
        all_recall_at_k = {cutoff: [] for cutoff in self.k}

        num_queries = 0
        num_positives = []
        num_negatives = []
        latencies = []

        for instance in tqdm(self.samples, desc='Samples'):
            query = instance['query']
            positive = list(instance['positive'])
            negative = list(instance['negative'])
            docs = positive + negative
            is_relevant = [True]*len(positive) + [False]*len(negative)

            if len(positive) == 0 or len(negative) == 0:
                continue

            num_queries += 1
            num_relevant = len(positive)
            num_positives.append(len(positive))
            num_negatives.append(len(negative))

            t0 = time.perf_counter()
            model_input = [[query, doc] for doc in docs]
            pred_scores = model.predict(model_input, convert_to_numpy=True, show_progress_bar=False)
            pred_scores_argsort = np.argsort(-pred_scores)  # Sort in decreasing order
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

            for cutoff in self.k:
                mrr_score = 0
                for rank, index in enumerate(pred_scores_argsort[0:cutoff]):
                    if is_relevant[index]:
                        mrr_score = 1 / (rank + 1)
                        break
                all_mrr_scores[cutoff].append(mrr_score)

                num_relevant_retrieved = 0
                for rank, index in enumerate(pred_scores_argsort[:cutoff]):
                    if is_relevant[index]:
                        num_relevant_retrieved += 1

                recall_at_cutoff = num_relevant_retrieved / num_relevant
                all_recall_at_k[cutoff].append(recall_at_cutoff)

            num_relevant_retrieved = 0
            for rank, index in enumerate(pred_scores_argsort[:num_relevant]):
                if is_relevant[index]:
                    num_relevant_retrieved += 1

            r_precision = num_relevant_retrieved / num_relevant
            all_r_precisions.append(r_precision)

        for cutoff in self.k:
            mrr_score = np.mean(all_mrr_scores[cutoff])
            recall_at_cutoff = np.mean(all_recall_at_k[cutoff])

            self.log_callback(epoch, steps, f'{self.name}/mrr@{cutoff}', mrr_score)
            self.log_callback(epoch, steps, f'{self.name}/recall@{cutoff}', recall_at_cutoff)

            logger.info(f"MRR@{cutoff}: {mrr_score * 100:.2f}")
            logger.info(f"Recall@{cutoff}: {recall_at_cutoff * 100:.2f}")

        r_precision = np.mean(all_r_precisions)
        self.log_callback(epoch, steps, f'{self.name}/r-precision', r_precision)
        logger.info(f"R-Precision: {r_precision * 100:.2f}")

        latency = np.mean(latencies) * 1000
        self.log_callback(epoch, steps, 'latency (ms/q)', latency)
        logger.info(f"Avg. latency (ms/query): {latency}")

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)
                for cutoff in self.k:
                    mrr_score = np.mean(all_mrr_scores[cutoff])
                    recall_at_cutoff = np.mean(all_recall_at_k[cutoff])
                    writer.writerow([epoch, steps, cutoff, mrr_score, recall_at_cutoff, r_precision])

        return np.mean(all_recall_at_k[10])
