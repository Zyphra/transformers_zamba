# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing suite for the PyTorch Zamba model."""

import math
import tempfile
import unittest

import pytest
from parameterized import parameterized

from transformers import AutoTokenizer, Zamba2Config, is_torch_available
from transformers.testing_utils import (
    require_bitsandbytes,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        Zamba2ForCausalLM,
        Zamba2ForSequenceClassification,
        Zamba2Model,
    )
    from transformers.models.zamba2.modeling_zamba2 import (
        Zamba2HybridDynamicCache,
    )


class Zamba2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=16,
        mamba_d_state=2,
        chunk_size=8,
        mamba_dt_rank="auto",
        num_hidden_layers=2,
        num_attention_heads=2,
        n_mamba_heads=8,
        mamba_ngroups=8,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_mamba_act="silu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
        layers_block_type=["mamba", "hybrid"],
        num_mem_blocks=1,
        use_mem_rope=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mamba_dt_rank = mamba_dt_rank
        self.mamba_d_state = mamba_d_state
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_mamba_heads = n_mamba_heads
        self.mamba_ngroups = mamba_ngroups
        self.chunk_size = chunk_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_mamba_act = hidden_mamba_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.layers_block_type = layers_block_type
        self.num_mem_blocks = num_mem_blocks
        self.use_mem_rope = use_mem_rope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return Zamba2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            mamba_dt_rank=self.mamba_dt_rank,
            mamba_d_state=self.mamba_d_state,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            n_mamba_heads=self.n_mamba_heads,
            intermediate_size=self.intermediate_size,
            chunk_size=self.chunk_size,
            hidden_act=self.hidden_act,
            mamba_ngroups=self.mamba_ngroups,
            hidden_mamba_act=self.hidden_mamba_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=True,
            initializer_range=self.initializer_range,
            use_mamba_kernels=False,
            layers_block_type=self.layers_block_type,
            num_mem_blocks=self.num_mem_blocks,
            use_mem_rope=self.use_mem_rope,
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        config.is_decoder = True

        return (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
        model = Zamba2Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = Zamba2ForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids, labels=token_labels)
        result = model(input_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        config.is_decoder = True
        config.add_cross_attention = False
        model = Zamba2ForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        # Attention: Zamba2 needs the cache to be initialized to return a cache!
        past_key_values = Zamba2HybridDynamicCache(config, input_ids.shape[0], model.dtype, device=model.device)
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 1), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            cache_position=torch.arange(
                input_ids.shape[1], input_ids.shape[1] + next_tokens.shape[1], device=model.device
            ),
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = Zamba2ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class Zamba2ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    test_torchscript = False
    all_model_classes = (
        (
            Zamba2Model,
            Zamba2ForCausalLM,
            Zamba2ForSequenceClassification,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (Zamba2ForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": Zamba2Model,
            "text-classification": Zamba2ForSequenceClassification,
            "text-generation": Zamba2ForCausalLM,
            "zero-shot": Zamba2ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = Zamba2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Zamba2Config, hidden_size=37)

    @unittest.skip("position_ids cannot be used to pad due to Mamba2 layers")
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_casual_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_initialization(self):
        r"""
        Overriding the test_initialization test as the A_log and D params of the Mamba block are initialized differently
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if "A_log" in name:
                        A = torch.arange(1, config.n_mamba_heads + 1, dtype=torch.float32)[None, :]
                        self.assertTrue(torch.allclose(param.data, torch.log(A), atol=1e-5, rtol=1e-5))
                    elif "D" in name:
                        # check if it's a ones like
                        self.assertTrue(torch.allclose(param.data, torch.ones_like(param.data), atol=1e-5, rtol=1e-5))
                    elif "dt_bias" in name:
                        dt = torch.exp(
                            torch.tensor([0, 1]) * (math.log(config.time_step_max) - math.log(config.time_step_min))
                            + math.log(config.time_step_min)
                        ).clamp(min=config.time_step_floor)
                        inv_dt = dt + torch.log(-torch.expm1(-dt))
                        if param.requires_grad:
                            self.assertTrue(param.data.max().item() <= inv_dt[1])
                            self.assertTrue(param.data.min().item() >= inv_dt[0])
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def test_mismatched_shapes_have_properly_initialized_weights(self):
        r"""
        Overriding the test_mismatched_shapes_have_properly_initialized_weights test because A_log and D params of the
        Mamba block are initialized differently and we tested that in test_initialization
        """
        self.skipTest("Cumbersome and redundant for Zamba2")

    def test_attention_outputs(self):
        r"""
        Overriding the test_attention_outputs test as the Zamba2 model outputs attention only for its attention layers
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions

            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )

    def _get_input_ids_and_config(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        return config, input_ids, input_mask

    def test_left_padding_compatibility(self):
        r"""
        Overriding the test_left_padding_compatibility test as the mamba layers accentuate the numerical differences
        effect of the left padding discussed in the issue in the note. Using a more permissive tolerance value.
        """
        import inspect
        # NOTE: left-padding results in small numerical differences. This is expected.
        # See https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535

        # First, filter out models that don't support left padding - generative and decoder-only.
        # Zamba2 is a decoder-only architecture
        decoder_only_classes = self.all_generative_model_classes

        # Then, test left-padding
        def _prepare_model_kwargs(input_ids, attention_mask, signature):
            model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "position_ids" in signature:
                position_ids = torch.cumsum(attention_mask, dim=-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                model_kwargs["position_ids"] = position_ids
            if "cache_position" in signature:
                cache_position = torch.arange(input_ids.shape[-1], device=torch_device)
                model_kwargs["cache_position"] = cache_position
            return model_kwargs

        for model_class in decoder_only_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()
            model = model_class(config).to(torch_device).eval()
            signature = inspect.signature(model.forward).parameters.keys()

            # Without padding
            model_kwargs = _prepare_model_kwargs(input_ids, attention_mask, signature)
            next_logits_wo_padding = model(**model_kwargs).logits[:, -1, :]

            # With left-padding (length 32)
            pad_size = (input_ids.shape[0], 32)
            padding = torch.ones(pad_size, dtype=input_ids.dtype, device=torch_device) * config.pad_token_id
            padded_input_ids = torch.cat((padding, input_ids), dim=1)
            padded_attention_mask = torch.cat((torch.zeros_like(padding), attention_mask), dim=1)
            model_kwargs = _prepare_model_kwargs(padded_input_ids, padded_attention_mask, signature)
            next_logits_with_padding = model(**model_kwargs).logits[:, -1, :]

            # They should result in very similar logits
            self.assertTrue(torch.allclose(next_logits_wo_padding, next_logits_with_padding, atol=3e-3))

    @require_flash_attn
    @require_torch_gpu
    @require_bitsandbytes
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_fp32_ln(self):
        r"""
        Overriding the test_flash_attn_2_fp32_ln test as the Zamba2 model, like Mixtral, doesn't support
        right padding + use cache with FA2
        """
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                dummy_input = inputs_dict[model.main_input_name]
                dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))
                # NOTE: Zamba2 does not support right padding + use_cache with FA2.
                dummy_attention_mask[:, -1] = 1

                model = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    low_cpu_mem_usage=True,
                    load_in_4bit=True,
                )

                for _, param in model.named_parameters():
                    # upcast only layer norms
                    if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                        param.data = param.data.to(torch.float32)

                _ = model(dummy_input)
                # with attention mask
                _ = model(dummy_input, attention_mask=dummy_attention_mask)

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        r"""
        Overriding the test_flash_attn_2_inference_padding_right test as the Zamba2 model, like Mixtral, doesn't support
        right padding + use cache with FA2
        """
        self.skipTest(reason="Zamba2 flash attention does not support right padding")

    @unittest.skip(reason="Zamba2 has its own special cache type")
    @parameterized.expand([(1, False), (1, True), (4, False)])
    def test_new_cache_format(self, num_beams, do_sample):
        pass


@require_torch
class Zamba2ModelIntegrationTest(unittest.TestCase):
    model = None
    tokenizer = None

    @classmethod
    @slow
    def setUpClass(cls):
        model_id = "Zyphra/Zamba2-1.2B"
        cls.model = Zamba2ForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True, revision="PR"
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id, revision="PR")

    @slow
    def test_simple_generate(self):
        self.model.to(torch_device)

        input_ids = self.tokenizer("Hey how are you doing on this lovely evening?", return_tensors="pt")[
            "input_ids"
        ].to(torch_device)
        out = self.model.generate(input_ids, do_sample=False, max_new_tokens=10)
        output_sentence = self.tokenizer.decode(out[0, :])
        self.assertEqual(
            output_sentence,
            "<s> Hey how are you doing on this lovely evening?\n\nI'm doing well, thanks for",
        )

        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits.to(dtype=torch.float32)

        EXPECTED_LOGITS_NO_GRAD = torch.tensor(
            [
               -5.9587, 10.5152,  7.0382, -2.8728, -4.8143, -4.8142, -4.8142, -4.8144,
               -4.8143, -4.8143, -4.8142, -4.8142,  6.0185, 18.0037, -4.8142, -4.8144,
               -4.8143, -4.8142, -4.8143, -4.8143, -4.8143, -4.8143, -4.8142, -4.8143,
               -4.8144, -4.8143, -4.8143, -4.8141, -4.8142, -4.8142, -4.8142, -4.8144,
               -4.8143, -4.8143, -4.8143, -4.8142, -4.8144, -4.8144, -4.8142, -4.8142
            ]
            , dtype=torch.float32)  # fmt: skip
        torch.testing.assert_close(logits[0, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD, rtol=1e-3, atol=1e-3)

    @slow
    def test_simple_batched_generate_with_padding(self):
        self.model.to(torch_device)

        inputs = self.tokenizer(
            ["Hey how are you doing on this lovely evening?", "When did the Roman empire "],
            padding=True,
            return_tensors="pt",
        ).to(torch_device)
        out = self.model.generate(**inputs, do_sample=False, max_new_tokens=10)
        output_sentences = self.tokenizer.batch_decode(out)
        self.assertEqual(
            output_sentences[0],
            "<s> Hey how are you doing on this lovely evening?\n\nI'm doing well, thanks for",
        )

        self.assertEqual(
            output_sentences[1],
            "[PAD][PAD][PAD][PAD]<s> When did the Roman empire 1st fall?\nThe Roman Empire fell in",
        )

        with torch.no_grad():
            logits = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits.to(
                dtype=torch.float32
            )

        EXPECTED_LOGITS_NO_GRAD_0 = torch.tensor(
            [
                -5.9611, 10.5208,  7.0411, -2.8743, -4.8167, -4.8167, -4.8167, -4.8168,
                -4.8167, -4.8167, -4.8167, -4.8166,  6.0218, 18.0062, -4.8167, -4.8168,
                -4.8167, -4.8167, -4.8167, -4.8168, -4.8168, -4.8168, -4.8167, -4.8167,
                -4.8168, -4.8167, -4.8167, -4.8165, -4.8167, -4.8167, -4.8167, -4.8169,
                -4.8168, -4.8168, -4.8168, -4.8166, -4.8169, -4.8168, -4.8167, -4.8167
            ]
            , dtype=torch.float32)  # fmt: skip

        EXPECTED_LOGITS_NO_GRAD_1 = torch.tensor(
            [
               0.1966,  6.3449,  3.8350, -5.7291, -6.5106, -6.5104, -6.5103, -6.5104,
               -6.5103, -6.5104, -6.5106, -6.5105,  7.8700, 13.5434, -6.5104, -6.5096,
               -6.5106, -6.5102, -6.5106, -6.5106, -6.5105, -6.5106, -6.5104, -6.5106,
               -6.5105, -6.5106, -6.5106, -6.5113, -6.5102, -6.5105, -6.5108, -6.5105,
               -6.5104, -6.5106, -6.5106, -6.5104, -6.5106, -6.5107, -6.5103, -6.5105          ]
            , dtype=torch.float32)  # fmt: skip

        torch.testing.assert_close(logits[0, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD_0, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(logits[1, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD_1, rtol=1e-3, atol=1e-3)