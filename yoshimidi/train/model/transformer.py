import torch

from yoshimidi.train.model.gpt import Gpt
from yoshimidi.train.model.gpt2 import Gpt2
from yoshimidi.train.model.transformer_config import TransformerConfig


def load_model(transformer_config: TransformerConfig) -> torch.nn.Module:
    """GPT-J implementation.

    - GPT-J uses GPT-3 architecture, but with:
      - Rotary Position Embeddings (RoPE).
      - Dense attention.
    - GPT-3 uses GPT-2 architecture, but with sparse attention (which we drop a la
    GPT-J).
    - GPT-2 uses GPT architecture, but with:
      - Layer normalization is moved to the input of each sub-block.
      - Layer normalization is added after the final self-attention block.
      - Weights of residual layers are scaled by a factor of 1/sqrt(N) where N is the
      number of residual layers.
    - GPT uses the original Transformer architecture, but with:
      - GELU activation instead of ReLU.

    TODO: Implement GPT-2/3/J architectures, currently only base GPT/Transformers are
    implemented.
    TODO: Use GELU activation.

    # ruff: noqa: E501
    Architecture links:
    - GPT-J: https://en.wikipedia.org/wiki/GPT-J#Architecture
    - GPT-3: https://arxiv.org/pdf/2005.14165.pdf#page=8
    - GPT-2: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=4
    - GPT: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
    - Transformer: https://arxiv.org/pdf/1706.03762.pdf
    """

    if transformer_config.type == "gpt":
        return Gpt(transformer_config)
    elif transformer_config.type == "gpt2":
        return Gpt2(transformer_config)
    else:
        raise NotImplementedError(
            f"Unknown transformer type: {transformer_config.type}"
        )
