from typing import Optional, List, Tuple, Union, Dict, Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BartConfig
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.utils import logging, ModelOutput

from src.utils.bart import BartForConditionalGeneration, BartModel, BartModelOutput, BartForConditionalGenerationOutput

logger = logging.get_logger(__name__)


class BartWithKnowledgeModel(BartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.g1 = nn.Sequential(
            nn.Linear(2 * self.config.d_model, self.config.d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.d_model, 1),
            nn.Sigmoid()
        )
        self.sent_id = 0

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            *,
            knowledge_embedding=None,
            init_input_embeds=None,
    ) -> Union[Tuple, BartModelOutput]:
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # Fuse knowledge and encoder transformed outputs
        encoder_last_hidden_state = encoder_outputs.last_hidden_state
        m = encoder_outputs.last_hidden_state.shape[1]
        if not self.training:
            knowledge_embedding = knowledge_embedding[self.sent_id].view(1, -1)
            init_input_embeds = torch.unsqueeze(init_input_embeds[self.sent_id], dim=0)
            self.sent_id = (self.sent_id + 1) % len(knowledge_embedding)

        knowledge_embedding = torch.unsqueeze(knowledge_embedding, dim=1).repeat(1, m, 1)
        input_fusion_head = torch.cat([init_input_embeds, knowledge_embedding], dim=2)

        # Knowledge relevance
        r = self.g1(input_fusion_head)
        rk_tilde = r * knowledge_embedding

        # Knowledge integration
        encoder_last_hidden_state += rk_tilde

        # Knowledge contribution
        ck = torch.norm(rk_tilde, dim=2)
        cx = torch.norm(encoder_outputs.last_hidden_state, dim=2)
        c = ck / (ck + cx)

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return BartModelOutput(
            knowledge_relevance=r,
            knowledge_contribution=c,
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class BartWithKnowledgeForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config, bart_cls=BartWithKnowledgeModel)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            *,
            knowledge_embedding=None,
            init_input_embeds=None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        # r"""
        # labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        #     Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        #     config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        #     (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        # Returns:
        # """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            knowledge_embedding=knowledge_embedding,
            init_input_embeds=init_input_embeds,
        )

        lm_logits = self.lm_head(outputs.last_hidden_state) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return BartForConditionalGenerationOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            knowledge_relevance=outputs.knowledge_relevance,
            knowledge_contribution=outputs.knowledge_contribution,
        )

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # !! Copied from GenerationMixin
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor

        # !! Custom Code starts here
        # Prevents knowledge_embedding from being sent to the encoder
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(
            **{k: v for k, v in encoder_kwargs.items() if k not in {'knowledge_embedding', 'init_input_embeds'}})

        return model_kwargs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {**super().prepare_inputs_for_generation(*args, **kwargs),
                **{'knowledge_embedding': kwargs['knowledge_embedding'],
                   'init_input_embeds': kwargs['init_input_embeds']}}
