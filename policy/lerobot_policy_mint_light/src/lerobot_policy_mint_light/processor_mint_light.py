#!/usr/bin/env python

from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.pipeline import ProcessorStepRegistry
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_mint_light import MINTLightConfig


@ProcessorStepRegistry.register(name="mint_light_tokenizer_processor")
class MINTLightTokenizerProcessorStep(TokenizerProcessorStep):
    """Tokenize SigLIP instructions while guaranteeing an attention mask."""

    def _tokenize_text(self, text: str | list[str]) -> dict[str, torch.Tensor]:
        tokenized = self.input_tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            padding_side=self.padding_side,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if "attention_mask" not in tokenized:
            input_ids = tokenized["input_ids"]
            pad_token_id = self.input_tokenizer.pad_token_id
            tokenized["attention_mask"] = (
                torch.ones_like(input_ids, dtype=torch.bool)
                if pad_token_id is None
                else input_ids.ne(pad_token_id)
            )
        return tokenized

def make_mint_light_pre_post_processors(
    config: MINTLightConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline]:
    """Build standard LeRobot normalization, text-tokenization and device pipelines."""

    features = {**(config.input_features or {}), **(config.output_features or {})}
    preprocessor = PolicyProcessorPipeline(
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            NormalizerProcessorStep(
                features=features, norm_map=config.normalization_mapping, stats=dataset_stats
            ),
            MINTLightTokenizerProcessorStep(
                tokenizer_name=config.language_model_name,
                max_length=config.tokenizer_max_length,
                padding_side="right",
                padding="max_length",
            ),
            DeviceProcessorStep(device=config.device),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline(
        steps=[
            UnnormalizerProcessorStep(
                features=config.output_features,
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
            DeviceProcessorStep(device="cpu"),
        ],
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor
