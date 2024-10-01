import sys
import unicodedata
from tqdm import tqdm
from typing import Callable
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, Adafactor, get_scheduler

from . import BaseModel, MmarcoReader

__all__ = ["SPLADE", "SPLADEv1", "SPLADEv2", "SPLADEplus", "SPLADEeff", "SPLADEv3"]


class SPLADE(BaseModel):
    """
    Class for the SPLADE model family.

    :param model_name_or_path: Path to the local model or the HF model ID.
    :param pooling: Sparse vector aggregation strategy (one of 'max' or 'sum').
    :param pruning_topk: Number of top tokens to keep from the activations.
    :param similarity: Similarity function to use for computing similarity scores (one of 'cos_sim' or 'dot_score').
    :param padding: Padding strategy. See https://huggingface.co/docs/transformers/pad_truncation.
    :param truncation: Truncation strategy. See https://huggingface.co/docs/transformers/pad_truncation.
    :param max_query_length: Maximum length of the query. Longer queries will be truncated.
    :param max_doc_length: Maximum length of the document. Longer documents will be truncated.
    :param add_special_tokens: Whether to add the special [CLS] andd [SEP] tokens to the input sequences.
    :param augment_query_to_maxlen: Whether to augment the query to the maximum query length with mask tokens.
    :param augment_doc_to_maxlen: Whether to augment the document to the maximum document length with mask tokens.
    :param query_prefix: Prefix to add to the query. Ideally, this should an unused token from the moel vocabulary.
    :param doc_prefix: Prefix to add to the document. Ideally, this should an unused token from the model vocabulary, different from the query prefix.
    :param freeze_layers_except_last_n: Freeze all model layers except the last n.
    :param device: Device to use for the model (one of 'cuda' or 'cpu').
    :param extra_files_to_load: List of extra files to load from HF hub.
    """
    def __init__(
        self,
        model_name_or_path: str,
        pooling: str = "max",
        pruning_topk: int = None,
        similarity: str = "cos_sim",
        padding: str = "longest",
        truncation: bool = True,
        do_lowercase: bool = False,
        max_query_length: int = 32,
        max_doc_length: int = 128,
        add_special_tokens: bool = True,
        augment_query_to_maxlen: bool = False,
        augment_doc_to_maxlen: bool = False,
        query_prefix: str = None,
        doc_prefix: str = None,
        freeze_layers_except_last_n: int = None,
        device: str = None,
        extra_files_to_load: list[str] = ["config_sparse_retrievers.json"],
    ) -> None:
        super().__init__(
            model_name_or_path=model_name_or_path,
            similarity=similarity,
            padding=padding,
            truncation=truncation,
            do_lowercase=do_lowercase,
            max_query_length=max_query_length,
            max_doc_length=max_doc_length,
            add_special_tokens=add_special_tokens,
            augment_query_to_maxlen=augment_query_to_maxlen,
            augment_doc_to_maxlen=augment_doc_to_maxlen,
            query_prefix=query_prefix,
            doc_prefix=doc_prefix,
            freeze_layers_except_last_n=freeze_layers_except_last_n,
            device=device,
            extra_files_to_load=extra_files_to_load,
        )
        assert pooling in ["max", "sum"], "The sparse vector aggregation strategy should either be 'max' or 'sum'."
        self.pooling = pooling
        self.pruning_topk = pruning_topk
        self.config = self._load_model_config()
        self.relu = torch.nn.ReLU().to(self.device)

    def forward(self, input_ids: torch.Tensor, input_masks: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SPLADE model.

        :param input_ids: A tensor of shape [batch_size, seq_length] containing the input token IDs.
        :param input_masks: A tensor of shape [batch_size, seq_length] containing the input token masks.
        :returns: A tensor of shape [batch_size, embedding_size].
        """
        out = self.model(input_ids=input_ids, attention_mask=input_masks)
        logits, attention_mask = out.logits, input_masks.unsqueeze(-1)

        if self.pooling == 'sum':
            activations = torch.sum(torch.log1p(self.relu(logits * attention_mask)), dim=1)
        else:
            activations = torch.amax(torch.log1p(input=self.relu(logits * attention_mask)), dim=1)

        if self.pruning_topk is not None:
            activations, _ = self._prune_activations(activations, keep_topk=self.pruning_topk)
        
        return activations
    
    def fit(
        self,
        dataloader: DataLoader = None,
        train_language: str = None,
        mmarco_data_config: dict[str, object] = None,
        rank_loss_config: dict[str, object] = None,
        reg_loss_config: dict[str, object] = {'query_reg': 'FlopsLoss', 'query_reg_weight': 3e-4, 'doc_reg': 'FlopsLoss', 'doc_reg_weight': 1e-4},
        steps: int = 150000,
        batch_size: int = 128,
        optimizer_name: str = 'AdamW',
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        scheduler: str = 'linear',
        warmup_ratio: float = 0.04,
        max_grad_norm: float = 1.0,
        output_path: str = None,
        ckpt_path: str = None,
        ckpt_save_steps: int = None,
        ckpt_save_limit: int = 0,
        fp16_amp: bool = False,
        bf16_amp: bool = False,
        log_every_n_steps: int = 0,
        log_callback: Callable[[int, int, int, float, float], None] = None,
    ):
        """
        Train a SPLADE model.

        :param dataloader: PyTorch DataLoader object for training data. If provided, will default over loading the mMARCO training data.
        :param train_language: Language to use with mMARCO. If a DataLoader is provided, this will be ignored.
        :param mmarco_data_config: Configuration for loading the mMARCO training data. If a DataLoader is provided, this will be ignored.
        :param rank_loss_config: Configuration for the ranking loss function.
        :param reg_loss_config: Configuration for the regularization loss functions.
        :param steps: Number of training steps.
        :param batch_size: Batch size for training.
        :param optimizer_name: Optimizer to use for training.
        :param learning_rate: Learning rate for the optimizer.
        :param weight_decay: Weight decay for the optimizer.
        :param scheduler: Learning rate scheduler.
        :param warmup_ratio: Warmup ratio for the scheduler.
        :param max_grad_norm: Used for gradient normalization.
        :param output_path: Path to save the trained model.
        :param ckpt_path: Path to save the checkpoints.
        :param ckpt_save_steps: Will save a checkpoint after so many steps.
        :param ckpt_save_limit: Total number of checkpoints to store.
        :param fp16_amp: Whether to use FP16 mixed precision training.
        :param bf16_amp: Whether to use BF16 mixed precision training.
        :param log_every_n_steps: Log training metrics every n steps.
        :param log_callback: Callback function for logging.
        """
        assert dataloader is not None or train_language is not None, "Either 'dataloader' or 'train_language' must be provided."
        assert rank_loss_config.get('name') in ["InfoNCELoss", "MarginMSELoss", "KLDLoss"], "The ranking loss function should be one of 'InfoNCE', 'MarginMSE', or 'KLD'."
        assert reg_loss_config.get('query_reg') in ["L1Loss", "FlopsLoss"], "The query regularization loss function should be one of 'L1' or 'FLOPS'."
        assert reg_loss_config.get('doc_reg') in ["L1Loss", "FlopsLoss"], "The document regularization loss function should be one of 'L1' or 'FLOPS'."
        assert optimizer_name in ["AdamW", "Adafactor"], "The optimizer should be one of 'AdamW' or 'Adafactor'."

        if dataloader is None:
            assert mmarco_data_config.get('training_sample_format') == "tuple_with_scores" if (rank_loss_config.get('name') == "MarginMSELoss" or rank_loss_config.get('name') == "KLDLoss") else True, (
                "The training sample format should be 'tuple_with_scores' for MarginMSE and KLD distillation losses."
            )
            data = MmarcoReader(
                lang=train_language, 
                load_train=True, 
                max_train_examples=steps * batch_size,
                **mmarco_data_config,
            ).load()
            dataloader = DataLoader(
                data['train'], 
                shuffle=True, 
                batch_size=batch_size, 
                collate_fn=lambda b: self.collate(b, negs_per_query=mmarco_data_config.get('negs_per_query', 1)),
            )
        data_iterator = iter(dataloader)
        batch_size = dataloader.batch_size

        if fp16_amp or bf16_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = filter(lambda x: x[1].requires_grad, self.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer_class = getattr(sys.modules[__name__], optimizer_name)
        optimizer_params = {"lr": learning_rate}
        if optimizer_name == "Adafactor":
            optimizer_params.update({'clip_threshold': 1.0, 'scale_parameter': False, 'relative_step': False, 'warmup_init': False})
        else:
            optimizer_params.update({'no_deprecation_warning': True, 'eps': 1e-7})
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = get_scheduler(optimizer=optimizer, name=scheduler, num_warmup_steps=int(steps * warmup_ratio), num_training_steps=steps)

        rank_loss_class = getattr(sys.modules[__name__], rank_loss_config.get('name'))
        rank_loss_params = {k: v for k, v in rank_loss_config.items() if k not in {'name', 'use_ib_negs'}}
        rank_loss = rank_loss_class(**rank_loss_params)

        query_reg_loss_class = getattr(sys.modules[__name__], reg_loss_config.get('query_reg'))
        query_reg_loss_params = {'weight': reg_loss_config.get('query_reg_weight'), **({'target_step': int(steps/3)} if reg_loss_config.get('query_reg') == 'FlopsLoss' else {})}
        query_reg_loss = query_reg_loss_class(**query_reg_loss_params)

        doc_reg_loss_class = getattr(sys.modules[__name__], reg_loss_config.get('doc_reg'))
        doc_reg_loss_params = {'weight': reg_loss_config.get('doc_reg_weight'), **({'target_step': int(steps/3)} if reg_loss_config.get('doc_reg') == 'FlopsLoss' else {})}
        doc_reg_loss = doc_reg_loss_class(**doc_reg_loss_params)

        self.model.zero_grad()
        self.model.train()

        skip_scheduler = False
        for global_step in tqdm(range(steps), desc="Iteration"):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                batch = next(data_iterator)
            batch = self._to_device(batch)

            fwd_pass_context = autocast(dtype=torch.bfloat16) if bf16_amp else (autocast(dtype=torch.float16) if fp16_amp else nullcontext())
            with fwd_pass_context:
                query_reps = self(batch['query_input_ids'], batch['query_input_masks']) #[bs,dim]
                pos_reps = self(batch['pos_input_ids'], batch['pos_input_masks']) #[bs,dim]
                neg_reps = self(batch['neg_input_ids'], batch['neg_input_masks']) #[bs*negs_per_query,dim]

                if query_reps.size(0) != batch_size:
                    continue # This is to avoid the last batch (w. size < bs) to have too much importance in the weights' update
                    
                negs_per_query = neg_reps.size(0) // batch_size
                if global_step == 0:
                    print(f"INFO: Training process uses {negs_per_query} hard negatives per query.")

                pos_scores = self.compute_pairwise_similarity(q_embs=query_reps, d_embs=pos_reps) #[bs]
                neg_scores = self.compute_pairwise_similarity(
                    q_embs=query_reps.unsqueeze(1).expand(-1, negs_per_query, -1), 
                    d_embs=neg_reps.view(batch_size, negs_per_query, -1),
                ) #[bs,negs_per_query]

                if rank_loss_config.get('name') == 'InfoNCELoss':
                    if rank_loss_config.get('use_ib_negs', False):
                        ib_sim_matrix = self.compute_batchwise_similarity(q_embs=query_reps, d_embs=pos_reps) #[bs,bs]
                        mask = torch.ones(ib_sim_matrix.size(), dtype=torch.bool).fill_diagonal_(0)
                        ib_sim_matrix = ib_sim_matrix[mask].view(batch_size, batch_size - 1) #[bs,bs-1]
                        neg_scores = torch.cat([neg_scores, ib_sim_matrix], dim=-1) #[bs,negs_per_query+bs-1]
                    
                    rank_loss_value = rank_loss(pos_scores=pos_scores, neg_scores=neg_scores)
                else:
                    rank_loss_value = rank_loss(
                        pos_scores=pos_scores, 
                        neg_scores=neg_scores, 
                        teacher_pos_scores=batch['target_pos_scores'], 
                        teacher_neg_scores=batch['target_neg_scores'].view(batch_size, negs_per_query),
                    )
                    if global_step == 0 and rank_loss_config.get('teacher_scale', 1.0) == 1.0:
                        student_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1).flatten()
                        teacher_scores = torch.cat([batch['target_pos_scores'], batch['target_neg_scores']], dim=0)
                        print(
                            f"INFO: Student scores from the first batch range between [{student_scores.min().item():.2f}, {student_scores.max().item():.2f}], " +
                            f"while teacher scores range between [{teacher_scores.min().item():.2f}, {teacher_scores.max().item():.2f}]. " +
                            f"Consider adjusting the 'teacher_scale' loss parameter (e.g., to '{(student_scores.max()/teacher_scores.max()).item():.2f}')."
                        )

                query_reg_loss_value = query_reg_loss(reps=query_reps, step=global_step)
                doc_reg_loss_value = doc_reg_loss(reps=torch.cat([pos_reps, neg_reps], dim=0), step=global_step)

                loss_value = rank_loss_value + query_reg_loss_value + doc_reg_loss_value

            if fp16_amp:
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

            if log_callback is not None and log_every_n_steps > 0 and global_step % log_every_n_steps == (log_every_n_steps - 1): 
                log_callback(0, 0, global_step, scheduler.get_last_lr()[0], loss_value.item(), "loss")
                log_callback(0, 0, global_step, scheduler.get_last_lr()[0], rank_loss_value.item(), "rank_loss")
                log_callback(0, 0, global_step, scheduler.get_last_lr()[0], doc_reg_loss_value.item(), "doc_reg_loss")
                log_callback(0, 0, global_step, scheduler.get_last_lr()[0], query_reg_loss_value.item(), "query_reg_loss")

            if ckpt_path and ckpt_save_steps is not None and ckpt_save_steps > 0 and global_step % ckpt_save_steps == 0:
                self._save_checkpoint(ckpt_path, ckpt_save_limit, global_step)

        self.save(output_path)

    def _prune_activations(self, activations: torch.Tensor, keep_topk: int) -> torch.Tensor:
        """
        Prune the sparse vectors to keep the top-k highest activation values only.

        :param activations: A tensor of shape [batch_size, vocab_size] containing the activations.
        :param keep_topk: Number of top tokens to keep from the activations.
        :returns: A tensor of shape [batch_size, vocab_size] and one of shape [batch_size, keep_topk] containing 
            the pruned activations and the corresponding top-k indices, respectively.
        """
        topk_values, topk_indices = torch.topk(activations, k=int(keep_topk), dim=1, largest=True, sorted=True)
        pruned_activations = torch.zeros_like(activations).scatter(dim=1, index=topk_indices, src=topk_values)
        return pruned_activations, topk_indices

    def decode(
        self, 
        activations: torch.Tensor, 
        output_type: str = 'tokens', # 'token_ids', 'words'
        clean_up_tokenization_spaces: bool = False, 
        skip_special_tokens: bool = True, 
        topk_tokens: int = 96,
    ) -> list[str]:
        """ 
        Decode the top-k activated tokens to text.
        
        :param activations: A tensor of shape [batch_size, vocab_size] containing the activations.
        :param clean_up_tokenization_spaces: Whether to ignore tokenization spaces when decoding.
        :param skip_special_tokens: Whether to ignore special tokens when decoding.
        :param topk_tokens: Number of top tokens to keep from the activations.
        :returns: Returns a list of decoded texts.
        """
        assert output_type in ['token_ids', 'tokens', 'words']
        scores, activations = torch.topk(input=activations, k=topk_tokens, dim=-1)
        bows = []
        for score, activation in zip(scores, activations):
            score = torch.round(score * 100).int()
            non_zero_indices = score.nonzero().squeeze(1)
            weights = score.index_select(dim=-1, index=non_zero_indices).tolist()
            tokens = activation.index_select(dim=-1, index=non_zero_indices).tolist()
            if output_type != 'token_ids':
                tokens = self.tokenizer.convert_ids_to_tokens(tokens) if output_type == 'tokens' else self.tokenizer.batch_decode(tokens)
                combined_dict = {}
                for token, weight in zip(tokens, weights):
                    norm_token = ''.join(c for c in unicodedata.normalize('NFD', token.lower()) if unicodedata.category(c) != 'Mn')
                    if norm_token in combined_dict:
                        combined_dict[norm_token] += weight
                    elif norm_token.endswith('s') and norm_token[:-1] in combined_dict:
                        combined_dict[norm_token[:-1]] += weight
                    elif norm_token + 's' in combined_dict:
                        combined_dict[norm_token] = combined_dict.pop(norm_token + 's') + weight
                    else:
                        combined_dict[norm_token] = weight
                bows.append(combined_dict)
            else:
                bows.append(dict(zip(tokens, weights)))
        return bows


        # return [
        #     " ".join(activation.translate(str.maketrans("", "", string.punctuation)).split())
        #     for activation in self.tokenizer.batch_decode(
        #         sequences=activations,
        #         clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        #         skip_special_tokens=skip_special_tokens,
        #     )
        # ]


class SPLADEv1(SPLADE):
    """
    SPLADE-sum model (2021-07). Reference: https://arxiv.org/abs/2107.05720
    """
    def __init__(self, **kwargs):
        defaults = {'pooling': 'sum'}
        self._override_defaults(defaults, kwargs)
        super().__init__(**defaults)

    def fit(self, **kwargs):
        defaults = {
            'mmarco_data_config': {
                'training_sample_format': 'triplet', 
                'negs_type': 'original',
            },
            'rank_loss_config': {
                'name': 'InfoNCELoss',
                'use_ib_negs': True,
                'temperature': 0.05,
            },
            'reg_loss_config': {
                'query_reg': 'FlopsLoss',
                'query_reg_weight': 3e-4,
                'doc_reg': 'FlopsLoss',
                'doc_reg_weight': 1e-4,
            },
            'optimizer_name': 'AdamW',
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'scheduler': 'linear',
            'warmup_ratio': 0.04,
        }
        self._override_defaults(defaults, kwargs)
        super().fit(**defaults)


class SPLADEv2(SPLADE):
    """
    SPLADE-max model (2021-09). Reference: https://arxiv.org/abs/2109.10086
    """
    def __init__(self, **kwargs):
        defaults = {'pooling': 'max'}
        self._override_defaults(defaults, kwargs)
        super().__init__(**defaults)

    def fit(self, **kwargs):
        defaults = {
            'mmarco_data_config': {
                'training_sample_format': 'triplet', 
                'negs_type': 'original',
            },
            'rank_loss_config': {
                'name': 'InfoNCELoss',
                'use_ib_negs': True,
                'temperature': 0.05,
            },
            'reg_loss_config': {
                'query_reg': 'FlopsLoss',
                'query_reg_weight': 3e-4,
                'doc_reg': 'FlopsLoss',
                'doc_reg_weight': 1e-4,
            },
            'optimizer_name': 'AdamW',
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'scheduler': 'linear',
            'warmup_ratio': 0.04,
        }
        self._override_defaults(defaults, kwargs)
        super().fit(**defaults)


class SPLADEplus(SPLADE):
    """
    DistilSPLADE-max model (2022-05). Reference: https://arxiv.org/abs/2205.04733
    """
    def __init__(self, **kwargs):
        defaults = {'pooling': 'max'}
        self._override_defaults(defaults, kwargs)
        super().__init__(**defaults)

    def fit(self, **kwargs):
        defaults = {
            'mmarco_data_config': {
                'training_sample_format': 'tuple_with_scores',
                'negs_type': 'hard', 
                'negs_mining_systems': 'bm25',
                'negs_per_query': 1,
            },
            'rank_loss_config': {
                'name': 'MarginMSELoss',
                'teacher_scale': 0.08,
            },
            'reg_loss_config': {
                'query_reg': 'FlopsLoss',
                'query_reg_weight': 3e-4,
                'doc_reg': 'FlopsLoss',
                'doc_reg_weight': 1e-4,
            },
            'optimizer_name': 'AdamW',
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'scheduler': 'linear',
            'warmup_ratio': 0.04,
        }
        self._override_defaults(defaults, kwargs)
        super().fit(**defaults)


class SPLADEplusEnsemble(SPLADE):
    """
    EnsembleDistilSPLADE-max model (2022-05). Reference: https://arxiv.org/abs/2205.04733
    """
    def __init__(self, **kwargs):
        defaults = {'pooling': 'max'}
        self._override_defaults(defaults, kwargs)
        super().__init__(**defaults)

    def fit(self, **kwargs):
        defaults = {
            'mmarco_data_config': {
                'training_sample_format': 'tuple_with_scores',
                'negs_type': 'hard', 
                'negs_mining_systems': 'all',
                'negs_per_query': 1,
            },
            'rank_loss_config': {
                'name': 'MarginMSELoss',
                'teacher_scale': 0.08,
            },
            'reg_loss_config': {
                'query_reg': 'FlopsLoss',
                'query_reg_weight': 3e-4,
                'doc_reg': 'FlopsLoss',
                'doc_reg_weight': 1e-4,
            },
            'optimizer_name': 'AdamW',
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'scheduler': 'linear',
            'warmup_ratio': 0.04,
        }
        self._override_defaults(defaults, kwargs)
        super().fit(**defaults)


class SPLADEeff(SPLADE):
    """
    Efficient-SPLADE model (2022-07). Reference: https://arxiv.org/abs/2207.03834
    """
    def __init__(self, **kwargs):
        defaults = {'pooling': 'max'}
        self._override_defaults(defaults, kwargs)
        super().__init__(**defaults)

    def fit(self, **kwargs):
        defaults = {
            'mmarco_data_config': {
                'training_sample_format': 'tuple_with_scores',
                'negs_type': 'hard', 
                'negs_mining_systems': 'all',
                'negs_per_query': 1,
            },
            'rank_loss_config': {
                'name': 'KLDLoss',
            },
            'reg_loss_config': {
                'query_reg': 'L1Loss',
                'query_reg_weight': 1e-2,
                'doc_reg': 'FlopsLoss',
                'doc_reg_weight': 1e-4,
            },
            'optimizer_name': 'AdamW',
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'scheduler': 'linear',
            'warmup_ratio': 0.04,
        }
        self._override_defaults(defaults, kwargs)
        super().fit(**defaults)


class SPLADEv3(SPLADE):
    """
    SPLADEv3 model (2024-03). Reference: https://arxiv.org/abs/2403.06789
    """
    def __init__(self, **kwargs):
        defaults = {'pooling': 'max'}
        self._override_defaults(defaults, kwargs)
        super().__init__(**defaults)

    def fit(self, **kwargs):
        defaults = {
            'mmarco_data_config': {
                'training_sample_format': 'tuple_with_scores',
                'negs_type': 'hard', 
                'negs_mining_systems': 'all',
                'negs_per_query': 8,
            },
            'rank_loss_config': {
                'name': 'KLDLoss',
            },
            'reg_loss_config': {
                'query_reg': 'FlopsLoss',
                'query_reg_weight': 3e-4,
                'doc_reg': 'FlopsLoss',
                'doc_reg_weight': 1e-4,
            },
            'optimizer_name': 'AdamW',
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'scheduler': 'linear',
            'warmup_ratio': 0.04,
        }
        self._override_defaults(defaults, kwargs)
        super().fit(**defaults)
