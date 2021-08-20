"""Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause"""

import torch
from transformers import PreTrainedModel, BartModel, BartConfig, BartPretrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput, BaseModelOutput
from torch import nn
import train_seq2seq_utils
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder
import torch.nn.functional as F
from generation_utils_multi_heads import GenerationMixinCustom


class BartModelTopic(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.num_decoder_layers_shared = None

        self.init_weights()

    # unchanged
    def get_input_embeddings(self):
        return self.shared

    # unchanged
    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # unchanged
    def get_encoder(self):
        return self.encoder

    # unchanged
    def get_decoder(self):
        return self.decoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            topic_ids=None,
            topic_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            topic_encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            use_mixed=False,
            use_head=0,
    ):
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = train_seq2seq_utils.shift_tokens_right(
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

            topic_encoder_outputs = self.encoder(
                input_ids=topic_ids,
                attention_mask=topic_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        encoder_outputs_combined = torch.cat([encoder_outputs[0], topic_encoder_outputs[0]], dim=1)
        attention_mask_combined = torch.cat([attention_mask, topic_mask], dim=1)

        decoder_args = {'input_ids': decoder_input_ids,
                        'attention_mask': decoder_attention_mask,
                        'encoder_hidden_states': encoder_outputs_combined,
                        'encoder_attention_mask': attention_mask_combined,
                        'head_mask': decoder_head_mask,
                        'cross_attn_head_mask': cross_attn_head_mask,
                        'past_key_values': past_key_values,
                        'inputs_embeds': decoder_inputs_embeds,
                        'use_cache': use_cache,
                        'output_attentions': output_attentions,
                        'output_hidden_states': True,
                        'return_dict': return_dict}

        decoder_outputs = self.decoder(**decoder_args)

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions
        )


class ConditionalGenerationCustomBartTopic(GenerationMixinCustom, BartPretrainedModel):
    base_model_prefix = "model"
    authorized_missing_keys = [r"final_logits_bias", r"encoder\.version", r"decoder\.version"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = BartModelTopic(config)
        self.model = base_model
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

    def forward(
            self,
            input_ids,
            attention_mask=None,
            topic_ids=None,
            topic_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            topic_encoder_outputs=None,
            past_key_values=None,
            lm_labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            generate=True,
            use_mixed=False,
            use_head=0,
            gate=None,
            use_gate_supervision=False,
            gate_prob=None,
            reward=None,
            use_sentence_gate_supervision=False,
            sent_gate=None,
            **unused,
    ):
        if "lm_labels" in unused:
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_args = {'input_ids': input_ids,
                      'attention_mask': attention_mask,
                      'decoder_input_ids': decoder_input_ids,
                      'encoder_outputs': encoder_outputs,
                      'topic_encoder_outputs': topic_encoder_outputs,
                      'decoder_attention_mask': decoder_attention_mask,
                      'past_key_values': past_key_values,
                      'use_cache': use_cache,
                      'output_attentions': output_attentions,
                      'output_hidden_states': output_hidden_states,
                      'return_dict': return_dict,
                      'topic_ids': topic_ids,
                      'topic_mask': topic_mask}

        outputs = self.model.forward(**input_args)
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        lm_logits = F.log_softmax(lm_logits, dim=-1)  # TODO: This is not logits, rename

        masked_lm_loss = None
        if not generate:
            lm_labels = train_seq2seq_utils.shift_tokens_left(decoder_input_ids, 1)
            loss_fct = nn.NLLLoss(ignore_index=1)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))

        return Seq2SeqLMOutput(
                            loss=masked_lm_loss,
                            logits=lm_logits,
                            past_key_values=outputs.past_key_values,
                            decoder_hidden_states=outputs.decoder_hidden_states,
                            decoder_attentions=outputs.decoder_attentions,
                            cross_attentions=outputs.cross_attentions,
                            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                            encoder_hidden_states=outputs.encoder_hidden_states,
                            encoder_attentions=outputs.encoder_attentions,
                            )

    def freeze_weights(self, num_decoder_layer_freeze=8):
        self.model.encoder.requires_grad_(False)
        self.model.num_decoder_layers_shared = num_decoder_layer_freeze
        for k in range(self.model.num_decoder_layers_shared):
            self.model.decoder.layers[k].requires_grad_(False)

    # unchanged
    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            topic_encoder_outputs=None,
            topic_mask=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "topic_encoder_outputs": topic_encoder_outputs,
            "topic_mask": topic_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    # unchanged
    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    # unchanged
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    # unchanged
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    # unchanged
    def get_decoder(self):
        return self.model.get_decoder()

    # unchanged
    def get_encoder(self):
        return self.model.get_encoder()

    # unchanged
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return train_seq2seq_utils.shift_tokens_right(labels, self.config.pad_token_id,
                                                      self.config.decoder_start_token_id)
