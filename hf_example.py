from transformers import AutoModel, AutoProcessor, AutoTokenizer
from PIL import Image
import requests
import torch
from transformers import TextStreamer

model = AutoModel.from_pretrained("/data2/omchat_dev/omchat/checkpoints/omchat-beta2_hf",trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()
processor = AutoProcessor.from_pretrained("/data2/omchat_dev/omchat/checkpoints/omchat-beta2_hf",trust_remote_code=True)

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt ="What's the content of the image?"
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, eos_token_id=model.generation_config.eos_token_id,  pad_token_id=processor.tokenizer.pad_token_id)

outputs = processor.tokenizer.decode(output_ids[0, inputs.input_ids.shape[1] :]).strip()
print (outputs)



