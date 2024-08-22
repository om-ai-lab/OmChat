# OmChat: A Family of Powerful Native Multimodal Language Models
We are thrilled to announce the release of OmChat Beta 2.0, a research version of our models from Om AI. This release includes the Qwen2 7B LLM-base and the InterVIT6B vision tower-based model, combining to form the OmChat Beta 13B model. These models are now available as open-source for researchers in the multimodal field, aimed at advancing meaningful research and contributing to the AI ecosystem's progress.

In the near future, we plan to release OmChat Beta 2.1, which will include support for long context as detailed in the OmChat paper, as well as a lighter version of the model. We will continue to update our latest versions for research purposes. For performance evaluation, we have tested our models using the OpenCompass benchmarks.

## Updates
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
pip install -e .
pip install flash-attn
```

### How to test a single image
```bash
python single_inference.py --model-path path-to-omchat-model --image-path path-to-image --question question-content
```

### Command-Line Interface
Execute conversational inference through the command line interface
```
python cli.py --model-path path-to-omchat-model --image-path path-to-image

```
Examples

```
User: how tall is he?
Asistant: The question is "how tall is he?". The question is asking about the height of the person in the picture. The answer is giving the height of the person in feet and inches. The person is 6 feet and 2 inches tall.
The answer is: 6'2"

```
### An Example with Huggingface transformers
Download huggingface model
```bash
git lfs install
git clone https://huggingface.co/omlab/omchat-v2.0-13B-single-beta_hf
```

```python
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from PIL import Image
import requests
import torch
from transformers import TextStreamer

model = AutoModel.from_pretrained("omlab/omchat-v2.0-13B-single-beta_hf",trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()
processor = AutoProcessor.from_pretrained("omlab/omchat-v2.0-13B-single-beta_hf", trust_remote_code=True)

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt ="What's the content of the image?"
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, eos_token_id=model.generation_config.eos_token_id,  pad_token_id=processor.tokenizer.pad_token_id)

outputs = processor.tokenizer.decode(output_ids[0, inputs.input_ids.shape[1] :]).strip()
print (outputs)
# The image features a stop sign in front of a Chinese archway, with a black car driving past. The stop sign is located on the left side of the scene, while the car is on the right side. There are also two statues of lions on either side of the archway, adding to the cultural ambiance of the scene.<|im_end|>

```

### Available HF Models from Om AI
- [omchat-v2.0-13B-single-beta_hf](https://huggingface.co/omlab/omchat-v2.0-13B-single-beta_hf) Currently, it supports only single images, but we will soon release models with multi-image and video support.


#### Model Comparison Results (less than 20B)

| Rank | Method                    | Avg Score |MMBench_V11|MMStar | MMMU_VAL | MathVista |
|------|---------------------------|-----------|----------|--------|----------|-----------|
| 1    | InternVL2-8B              | 64.1      | 79.4     | 61.5   | 51.2     | 58.3      |
| 2    | Omchat2.0-13B                 | 62.0      | 79.1     | 58.9   | 49.4     | 57.3      |
| 3    | InternLM-XComposer2.5     | 61.1      | 79.4     | 59.9   | 42.9     | 63.7      |
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


