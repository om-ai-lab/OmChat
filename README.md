# OmChat: A Family of Powerful Native Multimodal Language Models
We are thrilled to announce the release of OmChat Beta 2.0, a research version of our models from Om AI. This release includes the Qwen2 7B LLM-base and the InterVIT6B vision tower-based model, combining to form the OmChat Beta 13B model. These models are now available as open-source for researchers in the multimodal field, aimed at advancing meaningful research and contributing to the AI ecosystem's progress.

In the near future, we plan to release OmChat Beta 2.1, which will include support for long context as detailed in the OmChat paper, as well as a lighter version of the model. We will continue to update our latest versions for research purposes. For performance evaluation, we have tested our models using the OpenCompass benchmarks.

## Updates
* 09/10/2024: OmChat2.1-8B achieves top 1 on multi-image benchmark: Mantis-Eval for 8B models，which also outperformers GPT-4V. It also acheives SOTA on MMBench-Video for video benchmark.   🎉
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


#### Model Comparison Results on [OpenCompass](https://rank.opencompass.org.cn/leaderboard-multimodal/?m=REALTIME) (less than 20B models)

| Rank | Method                    | Avg Score |MMBench_V11|MMStar | MMMU_VAL | MathVista |
|------|---------------------------|-----------|----------|--------|----------|-----------|
| 1    | InternVL2-8B              | 64.1      | 79.4     | 61.5   | 51.2     | 58.3      |
| 2    | OmChat2.0-13B                 | 62.0      | 79.5  | 58.2   | 49.6     | 57.1      |
| 3    | InternLM-XComposer2.5     | 61.1      | 79.4     | 59.9   | 42.9     | 63.7      |
| 4    | InternVL2-4B              | 60.6      | 73.6     | 53.9   | 48.3     | 58.1      |
| 5    | GLM-4v-9B                 | 59.1      | 67.9     | 54.8   | 46.9     | 51.1      |
| 6    | InternLM-XComposer2-4     | 58.8      | 76.5     | 55.3   | 39.7     | 59.4      |
| 7    | MiniCPM-Llama3-V2.5       | 58.8      | 72       | 51.8   | 45.8     | 54.3      |
| 8    | WeMM                      | 58.3      | 75.7     | 57     | 45.3     | 54.9      |
| 9    | InternLM-XComposer2       | 57.1      | 77.6     | 56.2   | 41.4     | 59.5      |
| 10   | CogVLM2-19B-Chat          | 56.3      | 70.7     | 50.5   | 42.6     | 38.6      |


Model Comparison Results on [Mantis-Eval](https://huggingface.co/datasets/TIGER-Lab/Mantis-Eval) (8B models)

| Models          | Model size | Mantis-Eval |
| --------------- | ---------- |-------------|
| OmChat-2.1-8B   | 8B         | 67.28       |
| LLaVA OneVision | 7B         | 64.20       |
| GPT-4V          | \-         | 62.67       |
| Mantis-SigLIP   | 8B         | 59.45       |
| Mantis-Idefics2 | 8B         | 57.14       |
| Mantis-CLIP     | 8B         | 55.76       |
| VILA            | 8B         | 51.15       |
| Idefics2        | 8B         | 48.85       |


Model Comparison Results on [MMBench-Video](https://video-mme.github.io/home_page.html)

<table>
    <tr>
        <td rowspan="2">Model</td>
        <td rowspan="2">Frame</td>
        <td rowspan="2">Overall Mean</td>
        <td style="text-align: center;" colspan="5">Perception</td>
        <td style="text-align: center;" colspan="6">Reasoning</td>
    </tr>
    <tr>
        <td>CP</td>
        <td>FP-S</td>
        <td>FP-C</td>
        <td>HL</td>
        <td>Mean</td>
        <td>LR</td>
        <td>AR</td>
        <td>RR</td>
        <td>CSR</td>
        <td>TR</td>
        <td>Mean</td>
    </tr>
    <tr>
        <td>GPT-4o</td>
        <td>8</td>
        <td>1.62</td>
        <td>1.82</td>
        <td>1.59</td>
        <td>1.43</td>
        <td>1.95</td>
        <td>1.63</td>
        <td>1.33</td>
        <td>1.89</td>
        <td>1.60</td>
        <td>1.60</td>
        <td>1.44</td>
        <td>1.57</td>
    </tr>
    <tr>
        <td>GPT-4v</td>
        <td>8</td>
        <td>1.53</td>
        <td>1.68</td>
        <td>1.45</td>
        <td>1.43</td>
        <td>1.79</td>
        <td>1.51</td>
        <td>1.14</td>
        <td>1.81</td>
        <td>1.70</td>
        <td>1.59</td>
        <td>1.39</td>
        <td>1.52</td>
    </tr>
    <tr>
        <td>Gemini-Pro-v1.0</td>
        <td>8</td>
        <td>1.49</td>
        <td>1.72</td>
        <td>1.50</td>
        <td>1.28</td>
        <td>0.79</td>
        <td>1.49</td>
        <td>1.02</td>
        <td>1.66</td>
        <td>1.58</td>
        <td>1.59</td>
        <td>1.40</td>
        <td>1.45</td>
    </tr>
    <tr>
        <td>OmChat-2.1-8B</td>
        <td>32</td>
        <td>1.34</td>
        <td>1.54</td>
        <td>1.42</td>
        <td>1.12</td>
        <td>0.42</td>
        <td>1.37</td>
        <td>1.05</td>
        <td>1.40</td>
        <td>1.43</td>
        <td>1.37</td>
        <td>1.13</td>
        <td>1.27</td>
    </tr>
    <tr>
        <td>Gemini-Pro-v1.5</td>
        <td>8</td>
        <td>1.30</td>
        <td>1.51</td>
        <td>1.30</td>
        <td>0.98</td>
        <td>2.03</td>
        <td>1.32</td>
        <td>1.06</td>
        <td>1.62</td>
        <td>1.36</td>
        <td>1.25</td>
        <td>0.94</td>
        <td>1.22</td>
    </tr>
    <tr>
        <td>InternVL-Chat-v1.5</td>
        <td>8</td>
        <td>1.26</td>
        <td>1.51</td>
        <td>1.22</td>
        <td>1.01</td>
        <td>1.21</td>
        <td>1.25</td>
        <td>0.88</td>
        <td>1.4</td>
        <td>1.48</td>
        <td>1.28</td>
        <td>1.09</td>
        <td>1.22</td>
    </tr>
    <tr>
        <td>Claude-3v-Opus</td>
        <td>4</td>
        <td>1.19</td>
        <td>1.37</td>
        <td>1.11</td>
        <td>1.00</td>
        <td>1.56</td>
        <td>1.16</td>
        <td>1.12</td>
        <td>1.35</td>
        <td>1.36</td>
        <td>1.17</td>
        <td>1.05</td>
        <td>1.20</td>
    </tr>
    <tr>
        <td>mPLUG-Owl2</td>
        <td>8</td>
        <td>1.15</td>
        <td>1.34</td>
        <td>1.18</td>
        <td>0.99</td>
        <td>0.27</td>
        <td>1.15</td>
        <td>0.63</td>
        <td>1.33</td>
        <td>1.30</td>
        <td>1.03</td>
        <td>1.11</td>
        <td>1.11</td>
    </tr>
    <tr>
        <td>VideoStreaming</td>
        <td>64</td>
        <td>1.12</td>
        <td>1.38</td>
        <td>1.13</td>
        <td>0.8</td>
        <td>0.32</td>
        <td>1.13</td>
        <td>0.77</td>
        <td>1.27</td>
        <td>1.11</td>
        <td>1.01</td>
        <td>1.1</td>
        <td>1.09</td>
    </tr>
    <tr>
        <td>idefics2-8B</td>
        <td>8</td>
        <td>1.10</td>
        <td>1.23</td>
        <td>1.07</td>
        <td>0.89</td>
        <td>0.77</td>
        <td>1.06</td>
        <td>0.77</td>
        <td>1.27</td>
        <td>1.41</td>
        <td>1.11</td>
        <td>1.14</td>
        <td>1.16</td>
    </tr>
    <tr>
        <td>Video-LLaVA</td>
        <td>8</td>
        <td>1.05</td>
        <td>1.14</td>
        <td>1.08</td>
        <td>0.88</td>
        <td>0.50</td>
        <td>1.04</td>
        <td>0.72</td>
        <td>1.23</td>
        <td>1.03</td>
        <td>0.89</td>
        <td>0.97</td>
        <td>0.99</td>
    </tr>
</table>


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
🏠 [Github Repository](https://github.com/om-ai-lab/OmAgent)

🔆 [How to Evaluate the Generalization of Detection? A Benchmark for Comprehensive Open-Vocabulary Detection](https://arxiv.org/abs/2308.13177)(AAAI24)   
🏠 [Github Repository](https://github.com/om-ai-lab/OVDEval/tree/main)

🔆 [OmDet: Large-scale vision-language multi-dataset pre-training with multimodal detection network](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.12268)(IET Computer Vision)  
🏠 [Github Repository](https://github.com/om-ai-lab/OmDet)


