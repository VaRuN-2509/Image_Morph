import json
import requests
from io import BytesIO
from PIL import Image
import torch
import os
from aid_utils import load_im_from_path
from accelerate.utils import set_seed
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from torchvision import transforms

jsonl_path = "FreeMorph/sft.jsonl"

class Dataloader():
    def __init__(self,jsonl_path):
        self.jsonl_path = os.path.abspath(jsonl_path)

        self.image_pairs = {}
        self.model = None


    def read(self):
        BASE_URL = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/"
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for i,line in enumerate(f):
                entry = json.loads(line.strip())
                image_url = entry["open_image_input_url"]
                text_prompt = entry.get("text", "")
                output_img_url = entry["output_image"]

                if not (output_img_url.startswith("http://") or output_img_url.startswith("https://")):
                    output_img_url = BASE_URL + output_img_url.lstrip("/")

                image_pairs = {
                    "exp_id": i,
                    "input_image_url": image_url,
                    "output_image_url": output_img_url
                }

                # Optionally verify that images load before returning
                
                if self.model is None:
                    set_seed(42)
                    image_resolution = 768

                    self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

                    self.model = LlavaNextForConditionalGeneration.from_pretrained(
                        "llava-hf/llava-v1.6-mistral-7b-hf",
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                    )
                    self.model.to("cuda")

                shortener_prompt = (
                    "[INST] You are a helpful image editing assistant. "
                    "Rewrite the following edit instruction into a short, natural user-style request. "
                    "Keep the meaning and main intent, but remove redundant details and technical phrasing.\n\n"
                    f"{text_prompt}\n[/INST]"
                )

                inputs = self.processor(shortener_prompt, return_tensors="pt").to("cuda:0")
                output = self.model.generate(**inputs, max_new_tokens=50)
                edit_prompt = self.processor.decode(output[0], skip_special_tokens=True)
                image_pairs["edit_prompt"] = edit_prompt.split("[/INST]")[-1].strip()



                prompt = "[INST] <image>\nDescribe the image using five phrases and separate the phrases using commas.[/INST]"
                inputs = self.processor(
                    prompt,load_im_from_path(image_url) , return_tensors="pt"
                ).to("cuda:0")
                output = self.odel.generate(**inputs, max_new_tokens=50)
                prompt1 = self.processor.decode(output[0], skip_special_tokens=True)
                prompt1 = prompt1.split("[/INST]")[-1].strip()

                prompt = "[INST] <image>\nDescribe the image using five phrases and separate the phrases using commas.[/INST]"
                inputs = self.processor(
                    prompt, load_im_from_path(output_img_url), return_tensors="pt"
                ).to("cuda:0")
                output = self.model.generate(**inputs, max_new_tokens=50)
                prompt2 = self.processor.decode(output[0], skip_special_tokens=True)
                prompt2 = prompt2.split("[/INST]")[-1].strip()
                

                image_pairs["prompts"] = [prompt1, prompt2]
                    
                yield image_pairs  # returns one at a time

                        
                    