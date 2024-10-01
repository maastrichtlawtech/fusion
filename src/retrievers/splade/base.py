import os
import json
import shutil
from copy import deepcopy
from os.path import exists, join
from abc import ABC, abstractmethod

import torch
import heapq
import transformers
from huggingface_hub import hf_hub_download
from transformers.file_utils import default_cache_path
from transformers import AutoModelForMaskedLM, AutoTokenizer

from . import util

__all__ = ["BaseModel"]


class BaseModel(ABC, torch.nn.Module):
    """ 
    Class for the base siamese bi-encoder model from which all the neural models inherit.

    :param model_name_or_path: Path to the local model or the HF model ID.
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
        similarity: str = "cos_sim",
        padding: str = "longest",
        truncation: bool = True,
        do_lowercase: bool = False,
        max_query_length: int = 64,
        max_doc_length: int = 256,
        add_special_tokens: bool = True,
        augment_query_to_maxlen: bool = False,
        augment_doc_to_maxlen: bool = False,
        query_prefix: str = None,
        doc_prefix: str = None,
        freeze_layers_except_last_n: int = None,
        device: str = None,
        extra_files_to_load: list[str] = ["config_sparse_retrievers.json"],
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.similarity = similarity
        self.padding = padding
        self.truncation = truncation
        self.do_lowercase = do_lowercase
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.add_special_tokens = add_special_tokens
        self.augment_query_to_maxlen = augment_query_to_maxlen
        self.augment_doc_to_maxlen = augment_doc_to_maxlen
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.model_config_filename = "config_sparse_retrievers.json"

        assert self.similarity in ["cos_sim", "dot_score"], (
            "Similarity should be either 'cos_sim' or 'dot_score'."
        )
        assert self.padding in ["max_length", "longest", "do_not_pad"], (
            "Padding strategy should be either 'max_length', 'longest', or 'do_not_pad'."
        )
        assert self.model_config_filename in extra_files_to_load, (
            f"Make sure to try loading the library-specific configuration file f'{self.model_config_filename}'."
        )

        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, device=self.device)

        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path).to(self.device)
        self.model.config.output_hidden_states = True

        if freeze_layers_except_last_n is not None:
            self.freeze_layers(num_trainable_top_layers=freeze_layers_except_last_n)

        self.model_folder = self._get_local_model_folder()
        for f in extra_files_to_load:
            _ = self._load_filepath(model_name=model_name_or_path, filename=f)

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, input_masks: torch.Tensor) -> torch.Tensor:
        """Pytorch forward method."""
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Method to train the model."""
        pass

    def collate(self, batch: list[list], negs_per_query: int):
        """
        Method to collate a batch of training samples into tensors that can be fed to the model.

        :param batch: Batch of training samples that can come with the following format:
            1. [query, pos, neg]
            2. [query, pos, neg1, ..., negN]
            3. [query, [pos, score], [neg1, score], ..., [negN, score]]
        :returns: A dictionary with the tokenized inputs and distillation scores (if any).
        """
        queries, positives, pos_scores, all_negatives, all_neg_scores = [], [], [], [], []
        for sample in batch:
            query, *passages = sample
            passages = passages[:1 + negs_per_query+1] # 1 pos + N negs.
            try:
                passages, scores = zip(*passages)
                pos_scores.append(scores[0])
                all_neg_scores.extend(scores[1:])
            except:
                pass
            queries.append(query)
            positives.append(passages[0])
            all_negatives.extend(passages[1:])
        
        tok_queries = self.tokenize(texts=queries, query_mode=True)
        tok_positives = self.tokenize(texts=positives, query_mode=False)
        all_tok_negatives = self.tokenize(texts=all_negatives, query_mode=False)

        return {
            "query_input_ids": tok_queries["input_ids"], "query_input_masks": tok_queries["attention_mask"],
            "pos_input_ids": tok_positives["input_ids"], "pos_input_masks": tok_positives["attention_mask"],
            "neg_input_ids": all_tok_negatives["input_ids"], "neg_input_masks": all_tok_negatives["attention_mask"],
            "target_pos_scores": torch.tensor(pos_scores) if pos_scores else None,
            "target_neg_scores": torch.tensor(all_neg_scores) if all_neg_scores else None,
        }

    def tokenize(self, texts: list[str], query_mode: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Method to tokenize a list of text strings into their corresponding token IDs and attention masks.

        :param texts: List of text strings to encode.
        :param query_mode: Whether to encode queries or documents.
        :returns: Tuple with the output logits, hidden states, and attention masks.
        """
        prefix = self.query_prefix if query_mode else self.doc_prefix
        if prefix is not None:
            texts = [prefix + t for t in texts]
        if self.do_lowercase:
            texts = [t.lower() for t in texts]

        max_length = self.max_query_length if query_mode else self.max_doc_length
        tok = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=max_length,
            add_special_tokens=self.add_special_tokens,
            return_attention_mask=True,
            return_tensors="pt",
        )

        augment_to_maxlen = self.augment_query_to_maxlen if query_mode else self.augment_doc_to_maxlen
        if augment_to_maxlen:
            tok['input_ids'][tok['input_ids'] == self.tokenizer.pad_token_id] = self.tokenizer.mask_token_id
            tok['attention_mask'][tok['input_ids'] == self.tokenizer.mask_token_id] = 1
        return tok

    def compute_pairwise_similarity(self, q_embs: torch.Tensor, d_embs: torch.Tensor) -> torch.Tensor:
        """
        Compute the pairwise similarity scores between a batch of queries and their corresponding documents.

        :param q_embs: A tensor of shape [batch_size, embedding_size].
        :param d_embs: A tensor of shape [batch_size, embedding_size].
        :returns: A tensor of shape [batch_size] containing the similarity scores.
        """
        if self.similarity == "cos_sim":
            q_embs = torch.nn.functional.normalize(q_embs, p=2, dim=-1)
            d_embs = torch.nn.functional.normalize(d_embs, p=2, dim=-1)
        return torch.sum(q_embs * d_embs, axis=-1)

    def compute_batchwise_similarity(self, q_embs: torch.Tensor, d_embs: torch.Tensor) -> torch.Tensor:
        """ 
        Compute the batchwise similarity scores between a batch of queries and documents.

        :param q_embs: A tensor of shape [num_queries, embedding_size].
        :param d_embs: A tensor of shape [num_docs, embedding_size].
        :returns: A tensor of shape [num_queries, num_docs] containing the similarity scores.
        """
        if self.similarity == "cos_sim":
            q_embs = torch.nn.functional.normalize(q_embs, p=2, dim=-1)
            d_embs = torch.nn.functional.normalize(d_embs, p=2, dim=-1)
        return torch.mm(q_embs, d_embs.t())

    def search(
        self, 
        queries: list[str], 
        documents: list[str], 
        batch_size: int = 32, 
        query_chunk_size: int = 100,
        doc_chunk_size: int = 500000,
        topk: int = 10,
    ) -> list[dict[str, float]]:
        """
        Method to perform similarity search between a list of queries  and a list of documents.

        :param queries: List of queries.
        :param documents: List of documents.
        :param batch_size: Encoding batch size for both queries and documents.
        :param query_chunk_size: Processing batch size for queries.
        :param doc_chunk_size: Processing batch size for documents.
        :param topk: Number of top documents to retrieve.
        :returns: A list of dictionaries with the top-k documents for each query.
        """
        query_embeddings = self.encode(queries, query_mode=True, batch_size=batch_size)
        doc_embeddings = self.encode(documents, query_mode=False, batch_size=batch_size)

        queries_result_list = [[] for _ in range(len(query_embeddings))]
        for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
            for doc_start_idx in range(0, len(doc_embeddings), doc_chunk_size):
                scores = self.compute_batchwise_similarity(
                    q_embs=query_embeddings[query_start_idx : query_start_idx + query_chunk_size],
                    d_embs=doc_embeddings[doc_start_idx : doc_start_idx + doc_chunk_size],
                )
                scores_topk_values, scores_topk_indices = torch.topk(
                    scores, min(topk, len(scores[0])), dim=1, largest=True, sorted=False
                )
                scores_topk_values = scores_topk_values.cpu().tolist()
                scores_topk_indices = scores_topk_indices.cpu().tolist()

                for query_itr in range(len(scores)):
                    for sub_doc_id, score in zip(scores_topk_indices[query_itr], scores_topk_values[query_itr]):
                        doc_id = doc_start_idx + sub_doc_id
                        query_id = query_start_idx + query_itr
                        if len(queries_result_list[query_id]) < topk:
                            # heaqp tracks the quantity of the first element in the tuple
                            heapq.heappush(queries_result_list[query_id], (score, doc_id))
                        else:
                            heapq.heappushpop(queries_result_list[query_id], (score, doc_id))

        for query_id in range(len(queries_result_list)):
            for doc_itr in range(len(queries_result_list[query_id])):
                score, doc_id = queries_result_list[query_id][doc_itr]
                queries_result_list[query_id][doc_itr] = {"doc_id": doc_id, "score": score}
            queries_result_list[query_id] = sorted(queries_result_list[query_id], key=lambda x: x["score"], reverse=True)
        
        return queries_result_list

    @torch.no_grad()
    def encode(
        self, 
        sentences: list[str], 
        query_mode: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """ 
        End-to-end method to encode a corpus of sentences with the model to obtain the final sentence embeddings.
        
        :param sentences: List of texts to encode.
        :param query_mode: Whether to encode queries or documents.
        :param batch_size: the batch size used for the computation.
        :param convert_to_tensor: Whether the output should be one large tensor.
        :param show_progress_bar: Whether to output a progress bar when encode sentences.
        :param kwargs: Additional arguments to pass to the model and tokenizer.
        :returns: A 2D tensor/array with shape [num_sentences, embedding_size].
        """
        embeddings = []
        for batch in util.batchify(
            X=sentences, 
            batch_size=batch_size, 
            tqdm_bar=show_progress_bar, 
            tqdm_msg=f"Encoding {'queries' if query_mode else 'documents'}",
        ):
            inputs = self.tokenize(texts=batch, query_mode=query_mode)
            inputs = self._to_device(inputs)
            out = self(inputs['input_ids'], inputs['attention_mask'])
            embeddings.append(out)

        if convert_to_tensor:
            embeddings = torch.cat(embeddings, dim=0)
        else:
            embeddings = np.asarray([emb.cpu().numpy() for emb in embeddings])

        return embeddings

    def evaluate(self, evaluator, output_path: str = None, epoch: int = -1, steps: int = -1):
        """
        Method to evaluate the model using a given evaluator.
        
        :param evaluator: The evaluator to use.
        :param output_path: Output path to save the evaluation results.
        :param epoch: The current epoch number.
        :param steps: The current step number.
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        return evaluator(self, output_path=output_path, epoch=epoch, steps=steps)

    def freeze_layers(self, num_trainable_top_layers: int) -> None:
        """
        Method to freeze all but the last N layers of the model.

        :param num_trainable_top_layers: Number of tailing layers to keep trainable.
        """
        parameters = list(self.model.named_parameters())
        parameters.reverse()
        for idx, (name, param) in enumerate(parameters):
            if idx < num_trainable_top_layers - 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def save(self, path: str) -> None:
        """ 
        Method to save the model, its configuration, and the tokenizer.
        
        :param path: Output path.
        """
        os.makedirs(path, exist_ok=True)
        self._save_model_config(path)
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)

    def _save_model_config(self, path: str) -> None:
        """
        Method to save the model configuration to a JSON file.

        :param path: Output path.
        """
        self.config["__version__"] = {
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
        }
        with open(file=join(path, self.model_config_filename), mode="w") as file:
            json.dump(obj=self.config, fp=file, indent=4)

    def _get_local_model_folder(self) -> str:
        """
        Method to get the local model folder path.
        """
        if exists(path=self.model_name_or_path):
            # Path to local checkpoint
            return self.model_name_or_path
        else:
            # Path to HF checkpoint saved in cache folder
            model_folder = join(default_cache_path, f"models--{self.model_name_or_path}".replace("/", "--"), "snapshots")
            snapshot = os.listdir(model_folder)[-1]
            return join(model_folder, snapshot)

    def _load_filepath(self, model_name: str, filename: str) -> str:
        """
        Method to load a given file from the Hugging Face Hub if not already present locally.

        :param model_name: The model name on Hugging Face Hub.
        :param filename: The name of the file to load either locally or on HF Hub.
        :returns: The local path of the file.
        """
        filepath = join(self.model_folder, filename)
        if exists(path=filepath):
            return filepath
        try:
            return hf_hub_download(repo_id=model_name, filename=filename)
        except Exception:
            return

    def _load_model_config(
        self, 
        exclude: list[str] = ['tokenizer', 'device', 'model_folder', 'model_config_filename', 'training'],
    ) -> dict[str, object]:
        """
        Method to load the model configuration from the library-specific 'config_sparse_retrievers.json' file (if any).

        :param exclude: A list of model attributes to exclude when saving.
        :returns: The model configuration.
        """
        config = deepcopy(vars(self))
        config = {k: v for k, v in config.items() if not k.startswith('_') and k not in exclude}
        config_path = self._load_filepath(model_name=self.model_name_or_path, filename=self.model_config_filename)
        if config_path is not None:
            with open(config_path, mode="r") as file:
                config = json.load(fp=file)
            for key, value in config.items():
                if key != 'model_name_or_path' and hasattr(self, key) and getattr(self, key) != value:
                    print(f"WARNING: {self.__class__.__name__} instance is configured with '{key} == {getattr(self, key)}' while it was trained with '{key} == '{value}'.")
        return config

    def _save_checkpoint(self, path: str, save_total_limit: int, step: int):
        """
        Method used during moddel training to save a new checkpoint and delete old ones.

        :param path: Path to the checkpoint directory.
        :param save_total_limit: Maximum number of checkpoints to keep.
        """
        self.save(join(path, str(step)))
        if save_total_limit is not None and save_total_limit > 0:
            old_checkpoints = []
            for subdir in os.listdir(path):
                if subdir.isdigit():
                    old_checkpoints.append({"step": int(subdir), "path": join(path, subdir)})
            if len(old_checkpoints) > save_total_limit:
                old_checkpoints = sorted(old_checkpoints, key=lambda x: x["step"])
                shutil.rmtree(old_checkpoints[0]["path"])

    def _to_device(self, batch: dict[str, object]) -> dict[str, object]:
        """
        Method to move a batch of data to the model device.

        :param batch: The batch of data.
        :returns: The batch of data moved to the model device.
        """
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch

    def _override_defaults(self, defaults: dict, kwargs: dict) -> None:
        """
        Method to override in-place a dictionary of default parameters of a class or function while issuing a warning that the defaults have changed.

        :param defaults: The default values of the function or class.
        :param kwargs: The keyword arguments to override the default values.
        """
        for k, val in kwargs.items():
            if k in defaults and defaults[k] != val:
                print(f"WARNING: Changing default behavior '{k}' of {self.__class__.__name__} from {defaults[k]} to {val}")
            defaults[k] = val
