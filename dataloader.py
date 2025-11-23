import json
import os
import torch
import requests
from io import BytesIO
from PIL import Image
from accelerate.utils import set_seed
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class Dataloader:
    def __init__(self, jsonl_path, save_dir="./eval_results/freemorph"):
        self.jsonl_path = os.path.abspath(jsonl_path)
        self.save_dir = os.path.abspath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.model = None
        self.processor = None


    def safe_load_image(self, path):
        """Load an image from a URL or local path safely; return None on failure."""
        try:
            if path.startswith("http://") or path.startswith("https://"):
                response = requests.get(path, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(path).convert("RGB")
            return image
        except Exception as e:
            print(f"⚠️ Failed to load image from {path}: {e}")
            return None

    def generate_qwen_response(self, messages, max_new_tokens=80):
        """Generate model output for given messages."""
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)
        return output_text[0].strip()

    def read(self):
        BASE_URL = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/"
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            for i, entry in enumerate(data):
                # Skip completed entries
                
                image_url = entry.get("open_image_input_url", "")
                text_prompt = entry.get("text", "")
                output_img_url = entry.get("output_image", "")

                # Fix relative URLs
                if not (output_img_url.startswith("http://") or output_img_url.startswith("https://")):
                    output_img_url = BASE_URL + output_img_url.lstrip("/")

                image_pairs = {
                    "exp_id": i,
                    "input_image_url": image_url,
                    "output_image_url": output_img_url,
                    "text": text_prompt,
                }

                # --- Load model lazily once ---
                if self.model is None:
                    set_seed(42)
                    print("🔹 Loading Qwen2.5-VL-7B-Instruct model...")
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        "Qwen/Qwen2.5-VL-7B-Instruct",
                        torch_dtype="auto",
                        device_map="auto",
                    )
                    self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

                # --- Load images ---
                input_img = self.safe_load_image(image_url)
                output_img = self.safe_load_image(output_img_url)

                if input_img is None or output_img is None:
                    print(f"⚠️ Skipping entry {i} due to missing image(s).")
                    continue

                # --- 1️⃣ Edit prompt shortener ---
                shortener_prompt = (
                    "You are a helpful image editing assistant. "
                    "Rewrite the following edit instruction into a short, natural user-style request. "
                    "Keep the meaning and main intent, but remove redundant details and technical phrasing.\n\n"
                    f"{text_prompt}"
                )
                messages_shortener = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": shortener_prompt}],
                    }
                ]
                edit_prompt = self.generate_qwen_response(messages_shortener, max_new_tokens=50)
                image_pairs["edit_prompt"] = edit_prompt

                # --- 2️⃣ Description prompt ---
                describe_prompt = (
                    "Describe the image in five short sentences capturing: "
                    "1. What is happening, "
                    "2. The main subjects and their actions, "
                    "3. The environment and surroundings, "
                    "4. Colors, lighting, or mood, "
                    "5. Additional notable details. "
                    "Return the answer as a single line where each sentence is separated by a comma."
                )

                messages_img1 = [
                    {"role": "user", "content": [{"type": "image", "image": image_url},
                                                 {"type": "text", "text": describe_prompt}]}
                ]
                prompt1 = self.generate_qwen_response(messages_img1, max_new_tokens=80)

                messages_img2 = [
                    {"role": "user", "content": [{"type": "image", "image": output_img_url},
                                                 {"type": "text", "text": describe_prompt}]}
                ]
                prompt2 = self.generate_qwen_response(messages_img2, max_new_tokens=80)

                image_pairs["prompts"] = [prompt1, prompt2]


                # Save per-item result (optional)
                # save_path = os.path.join(self.save_dir, f"result_{i}.json")
                # with open(save_path, "w", encoding="utf-8") as out_f:
                #     json.dump(image_pairs, out_f, indent=2, ensure_ascii=False)

                # Mark as done and save progress

                yield image_pairs
