# OmChat: A Family of Powerful Native Multimodal Language Models
......

## 🗓️ Updates
* 08/10/2024: The OmChat open-source project has been unveiled. 🎉
* 07/06/2024: [The OmChat research paper has been published.](https://arxiv.org/abs/2407.04923)


## Models & Scripts

### Installation

#### 1. **Clone this repository and navigate to the OmChat folder:**
```bash
git clone https://github.com/om-ai-lab/OmChat.git
cd OmChat
```

#### 2. **Install the inference package:**
```bash
conda create -n omchat python=3.10 -y
conda activate omchat
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
```

### Evaluation
```bash
bash eval.sh MODEL_PATH all
```

### How to test
```bash
bash test.sh MODEL_PATH
```

### Available Models from Om AI
- [OmChat8B]()
- [OmChat15B]()


#### Model Comparison Results (less than 20B)

| Rank | Method                    | Avg Score |MMBench_V11|MMStar | MMMU_VAL | MathVista |
|------|---------------------------|-----------|----------|--------|----------|-----------|
| 1    | InternVL2-8B              | 64.1      | 79.4     | 61.5   | 51.2     | 58.3      |
| 2    | InternLM-XComposer2.5     | 61.1      | 79.4     | 59.9   | 42.9     | 63.7      |
| 3    | Omchat15B                 | 61.1      | 77.2     | 58.4   | 44.6     | 57.2      |
| 4    | InternVL2-4B              | 60.6      | 73.6     | 53.9   | 48.3     | 58.1      |
| 5    | GLM-4v-9B                 | 59.1      | 67.9     | 54.8   | 46.9     | 51.1      |
| 6    | InternLM-XComposer2-4     | 58.8      | 76.5     | 55.3   | 39.7     | 59.4      |
| 7    | MiniCPM-Llama3-V2.5       | 58.8      | 72       | 51.8   | 45.8     | 54.3      |
| 8    | WeMM                      | 58.3      | 75.7     | 57     | 45.3     | 54.9      |
| 9    | InternLM-XComposer2       | 57.1      | 77.6     | 56.2   | 41.4     | 59.5      |
| 10   | CogVLM2-19B-Chat          | 56.3      | 70.7     | 50.5   | 42.6     | 38.6      |


## Citation
If you find our repository beneficial, please cite our paper:
```bibtex
@article{zhao2024omchat,
  title={OmChat: A Recipe to Train Multimodal Language Models with Strong Long Context and Video Understanding},
  author={Zhao, Tiancheng and Zhang, Qianqian and Lee, Kyusong and Liu, Peng and Zhang, Lu and Fang, Chunxin and Liao, Jiajia and Jiang, Kelei and Ma, Yibo and Xu, Ruochen},
  journal={arXiv preprint arXiv:2407.04923},
  year={2024}
}
```

## Acknowledgement
The codebase and models are built upon the following projects:
- [LLaVA](https://github.com/haotian-liu/LLaVA) 
- [LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [InternVL2](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/)
- [MoE-LLaVA](https://github.com/PKU-YuanGroup/MoE-LLaVA)
- [Qwen2](https://github.com/QwenLM/Qwen2)


## Projects from Om AI Team
If you are intrigued by multimodal algorithms, large language models, and agent technologies, we invite you to delve deeper into our research endeavors:  
🔆 [OmAgent: A Multi-modal Agent Framework for Complex Video Understanding with Task Divide-and-Conquer](https://arxiv.org/abs/2406.16620)
🏠 [Github Repository](https://github.com/om-ai-lab/OmAgentn)

🔆 [How to Evaluate the Generalization of Detection? A Benchmark for Comprehensive Open-Vocabulary Detection](https://arxiv.org/abs/2308.13177)(AAAI24)   
🏠 [Github Repository](https://github.com/om-ai-lab/OVDEval/tree/main)

🔆 [OmDet: Large-scale vision-language multi-dataset pre-training with multimodal detection network](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.12268)(IET Computer Vision)  
🏠 [Github Repository](https://github.com/om-ai-lab/OmDet)


