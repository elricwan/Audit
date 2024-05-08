# Audit
This repository for the paper "Enhancing the Efficiency and Accuracy of Underlying Asset Reviews in Structured Finance: The Application of Multi-agent Framework" on [arxiv](https://arxiv.org/abs/2405.04294).

Structured finance, which involves restructuring diverse assets into securities like MBS, ABS, and CDOs, enhances capital market efficiency but presents significant due diligence challenges. This study explores the integration of artificial intelligence (AI) with traditional asset review processes to improve efficiency and accuracy in structured finance. Using both open-sourced and close-sourced large language models (LLMs), we demonstrate that AI can automate the verification of information between loan applications and bank statements effectively. While close-sourced models such as GPT-4 show superior performance, open-sourced models like LLAMA3 offer a cost-effective alternative. Dual-agent systems further increase accuracy, though this comes with higher operational costs. This research highlights AI's potential to minimize manual errors and streamline due diligence, suggesting a broader application of AI in financial document analysis and risk management.

A new dataset of 98 bank statement and loan statement are presented on [example/pdf](https://github.com/elricwan/Audit/tree/main/example/pdf), based on the template on [example/html](https://github.com/elricwan/Audit/tree/main/example/html). The method to generate new statements is also available on [pdf_from_html](https://github.com/elricwan/Audit/blob/main/example/agents/pdf_from_html.py).

## Quick Tour

To reproduce the analysis and results in the paper, please take a look at the [demo](https://github.com/elricwan/Audit/blob/main/example/demo.ipynb)

## Citation

```latex
@misc{wan2024enhancing,
      title={Enhancing the Efficiency and Accuracy of Underlying Asset Reviews in Structured Finance: The Application of Multi-agent Framework}, 
      author={Xiangpeng Wan and Haicheng Deng and Kai Zou and Shiqi Xu},
      year={2024},
      eprint={2405.04294},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
