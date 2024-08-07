from transformers import CLIPImageProcessor
from PIL import Image
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import TensorType
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict


class GPT4ImageProcessor(CLIPImageProcessor):
    def __init__(self, *args, block_size=336, **kwargs):
        """
        Custom Image Processor that extends CLIPImageProcessor.

        Args:
            block_size (int): The size to which larger images will be resized and the size of the blocks
                              for dividing larger images. Defaults to 336.
        """
        super().__init__(*args, **kwargs)
        #size = {"shortest_edge": block_size}, crop_size = {"height": block_size, "width": block_size}
        self.block_size = block_size
        

    def preprocess(self, images, return_tensors='pt'):
        if not isinstance(images, list):
            images = [images]

        processed_tensors = []
        for image in images:
                #if image.size[0] > self.block_size or image.size[1] > self.block_size:
                # Resize for overall view
                overall_view = image.resize((self.block_size, self.block_size), Image.BICUBIC)
                overall_view_feature = super().preprocess(overall_view)
                processed_tensors.append(torch.tensor(overall_view_feature['pixel_values']))

                # Process each block
                blocks = self.divide_into_blocks(image.resize((self.block_size*2, self.block_size*2), Image.BICUBIC))
                for block in blocks:
                    block_feature = super().preprocess(block)
                    processed_tensors.append(torch.tensor(block_feature['pixel_values']))
                #else:
                #image_feature = super().preprocess(image)
                #processed_tensors.append(torch.tensor(image_feature['pixel_values']))

        # Concatenate all processed tensors along the batch dimension
        return {'pixel_values': torch.cat(processed_tensors, dim=0)}

    def divide_into_blocks(self, pil_img):
        width, height = pil_img.size
        blocks = []
        for i in range(0, height, self.block_size):
            for j in range(0, width, self.block_size):
                block = pil_img.crop((j, i, min(j + self.block_size, width), min(i + self.block_size, height)))
                blocks.append(block)
        return blocks
if __name__ == "__main__":
    image_file = "unittest/test.jpg"
    image = Image.open(image_file).convert('RGB')
    x = GPT4ImageProcessor(block_size=336)
    #image_tensor = x.preprocess(image)
    image_tensor = x.preprocess(image, return_tensors='pt')['pixel_values']
    print(image_tensor.shape)