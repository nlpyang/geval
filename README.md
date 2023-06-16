# Code for paper "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment" [https://arxiv.org/abs/2303.16634]

## Experiments on SummEval dataset
### Evaluate fluency on SummEval dataset
```python .\gpt4_eval.py --prompt .\prompts\summeval\flu_detailed.txt --save_fp .\results\gpt4_flu_detailed.json --summeval_fp .\data\summeval.json --key XXXXX```

### Meta Evaluate the G-Eval results

```python .\meta_eval_summeval.py --input_fp .\results\gpt4_flu_detailed.json --dimension fluency```

## Prompts and Evaluation Results

Prompts used to evaluate SummEval are in prompts/summeval

G-eval results on SummEval are in results

