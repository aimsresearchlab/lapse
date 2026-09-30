# Data licenses

Code (`tools/`, `tests/`, scripts under `exploratory/`) is MIT; see `LICENSE`.

The LAPSE stimuli, manifests, and human and model labels are released under
CC-BY-4.0. The stimuli are synthetic: every name, company, institution,
project, and residence is invented.

## Model outputs

`traces/` and the trace files under `exploratory/` contain model responses.
Each response remains subject to the terms of the provider that produced it
(DeepSeek, OpenAI, Z.ai, Anthropic, Google, xAI, Alibaba, and open-weight
models served locally). Hosted calls went through OpenRouter or the
provider's own endpoint, as recorded in each trace.

## Third-party text is not included

The natural-text screens and the real-utterance writer test draw on WildChat,
LoCoMo, and LongMemEval. Those texts are not redistributed here. The screening
code is in `tools/screen_ecological.py` and `tools/screen_wildchat.py`;
regenerate the pools from the corpora under their own terms.
