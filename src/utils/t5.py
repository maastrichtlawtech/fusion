"""
Custom T5 and mT5 classes to be used for sequence classification tasks. Usage:
```
if isinstance(self.config, T5Config):
    clean_t5_config(self.config, model_type='t5')
    self.model = T5EncoderForSequenceClassification.from_pretrained(
        model_name, config=self.config, revision=revision, **automodel_args
    )
```
"""
import torch
from torch import nn
from typing import Optional, Union, Tuple

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.t5.modeling_t5 import T5Config, T5ClassificationHead, T5EncoderModel
from transformers.models.mt5.modeling_mt5 import MT5Config, MT5ClassificationHead, MT5EncoderModel


def clean_t5_config(config: Union[T5Config, MT5Config], model_type: str):
    assert model_type in ['t5', 'mt5']
    setattr(config, 'pooling_mode', 'mean')
    setattr(config, 'model_type', model_type)
    setattr(config, 'use_cache', False)
    setattr(config, 'is_encoder_decoder', False)
    setattr(config, 'num_decoder_layers', None)
    delattr(config, 'task_specific_params') if hasattr(config, 'task_specific_params') else None


class PoolLayer(nn.Module):
    """
    Pooling layer on top of the commputed token embeddings. 

    :param pooling_mode: The pooling strategy to use.
    """
    def __init__(self, pooling_mode: str):
        super().__init__()
        assert pooling_mode in ['first', 'mean', 'max'], f"ERROR: Unknown pooling strategy '{pooling_mode}'"
        self.pooling_mode = pooling_mode

    def forward(self, token_embeddings: torch.Tensor, attention_masks: torch.Tensor) -> torch.Tensor:
        """
        Compute the passage vector by pooling the token embeddings.

        :param token_embeddings: A 3D tensor of size [batch_size, seq_len, d_model].
        :param attention_masks: A 2D tensor of size [batch_size, seq_len].
        :returns: A 2D tensor of size [batch_size, d_model].
        """
        if self.pooling_mode == 'first':
            text_vectors = token_embeddings[:, 0, :]
        elif self.pooling_mode == 'max':
            # Set all values of the [PAD] embeddings to large negative values (so that they are never considered as maximum for a channel).
            attention_masks_expanded = attention_masks.unsqueeze(-1).expand(token_embeddings.size())
            token_embeddings[attention_masks_expanded == 0] = -1e+9 if token_embeddings.dtype == torch.float32 else -1e+4
            # Compute the maxima along the 'seq_length' dimension (-> Tensor[batch_size, d_model]).
            text_vectors = torch.max(token_embeddings, dim=1).values
        else:
            # Set all values of the [PAD] embeddings to zeros (so that they are not taken into account in the sum for a channel).
            attention_masks_expanded = attention_masks.unsqueeze(-1).expand(token_embeddings.size())
            token_embeddings[attention_masks_expanded == 0] = 0.0
            # Compute the means by first summing along the 'seq_length' dimension (-> Tensor[batch_size, d_model]).
            sum_embeddings = torch.sum(token_embeddings, dim=1)
            # Then, divide all values of a passage vector by the original passage length.
            sum_mask = attention_masks_expanded.sum(dim=1) # -> Tensor[batch_size, d_model] where each value is the length of the corresponding passage.
            sum_mask = torch.clamp(sum_mask, min=1e-7) # Make sure not to have zeros by lower bounding all elements to 1e-7.
            text_vectors = sum_embeddings / sum_mask # Divide each dimension by the sequence length.
        return text_vectors


class T5EncoderForSequenceClassification(T5EncoderModel):
    """
    T5 encoder for sequence classification tasks.

    :param config: The T5 configuration object.
    """
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.pool_layer = PoolLayer(config.pooling_mode)
        self.classification_head = T5ClassificationHead(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """
        Forward pass of the T5 encoder for sequence classification tasks.

        :param input_ids: The input token IDs.
        :param attention_mask: The attention mask.
        :param head_mask: The head mask.
        :param inputs_embeds: The input embeddings.
        :param labels: The target labels.
        :param output_attentions: Whether to output attentions.
        :param output_hidden_states: Whether to output hidden states.
        :param return_dict: Whether to return a dictionary.
        :returns: The logits for the classification task or a dictionary containing the outputs.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        loss = None

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = self.pool_layer(outputs.last_hidden_state, attention_mask)
        logits = self.classification_head(sequence_output)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MT5EncoderForSequenceClassification(MT5EncoderModel):
    """
    mT5 encoder for sequence classification tasks.

    :param config: The mT5 configuration object.
    """
    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.pool_layer = PoolLayer(config.pooling_mode)
        self.classification_head = MT5ClassificationHead(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """
        Forward pass of the mT5 encoder for sequence classification tasks.

        :param input_ids: The input token IDs.
        :param attention_mask: The attention mask.
        :param head_mask: The head mask.
        :param inputs_embeds: The input embeddings.
        :param labels: The target labels.
        :param output_attentions: Whether to output attentions.
        :param output_hidden_states: Whether to output hidden states.
        :param return_dict: Whether to return a dictionary.
        :returns: The logits for the classification task or a dictionary containing the outputs.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        loss = None

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = self.pool_layer(outputs.last_hidden_state, attention_mask)
        logits = self.classification_head(sequence_output)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
