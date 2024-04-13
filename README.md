# 1.5b-Trim
Testing the effectiveness of deep layers in 1.5b language models

Paper Inspirations:
- https://arxiv.org/abs/2403.17887
- https://arxiv.org/abs/2402.17764

Model arch:
- https://huggingface.co/1bitLLM/bitnet_b1_58-3B

Method:
- https://github.com/huggingface/transformers/issues/2483

Datasets:
- https://huggingface.co/datasets/cais/mmlu
- https://huggingface.co/datasets/google/boolq

Next Steps:
- run test script
    - verify dataset pre-processing
    - verify output layers
- run experiments
    - mmlu
    - boolq
    - formulate outputs