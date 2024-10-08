import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from util.vision_util import process_vision_info
import torch
from pprint import pprint
from tqdm import tqdm


class QwenVLModel:
    def __init__(self, model_dir):
        """
        Initialize the model and processor from the pretrained directory.

        Args:
            model_dir (str): Path to the pretrained model directory.
        """
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir, device_map="auto", torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained(model_dir, padding_side="left")

    def generate_caption(self, image_path):
        """
        Generate a caption for the given image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Generated caption for the image.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": "What would you caption the character in this picture?"},
                ],
            }
        ]

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in [messages]
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate the output caption
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]

    def process_folder(self, folder_path):
        """
        Process a folder of images and generate captions for each, saving them to corresponding .txt files.

        Args:
            folder_path (str): Path to the folder containing images.
        """
        for file_name in os.listdir(folder_path):
            if file_name.endswith(('.png', '.jpg')):
                image_path = os.path.join(folder_path, file_name)
                # Generate caption
                caption = self.generate_caption(image_path)
                # Save caption to a .txt file with the same name as the image
                caption_file = os.path.join(folder_path, file_name.rsplit('.', 1)[0] + '.txt')
                with open(caption_file, 'w', encoding='utf-8') as f:
                    f.write(caption)
                print(f"Saved caption for {file_name} as {caption_file}")


def main():
    model_dir = "/home/admin/finetune-Qwen2-VL/train_output_v2/20241007174723"
    qwen_model = QwenVLModel(model_dir)

    # image_folder = "/home/admin/finetune-Qwen2-VL/M2_GB"
    image_folder = "/home/admin/finetune-Qwen2-VL/karuga"
    qwen_model.process_folder(image_folder)
    # subfolders = [f.path for f in os.scandir(image_folder) if f.is_dir()]
    # for s in tqdm(subfolders):
    #     print(f'processing folder: {s}')
    #     qwen_model.process_folder(s)


if __name__ == "__main__":
    main()
