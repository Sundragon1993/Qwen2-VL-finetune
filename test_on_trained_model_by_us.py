from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from util.vision_util import process_vision_info
from pprint import pprint
import torch

model_dir = "/home/admin/finetune-Qwen2-VL/train_output_v2/20241004162153"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, device_map="auto", torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", adapter_name='adapter_model'
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-2B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
processor = AutoProcessor.from_pretrained(model_dir, padding_side="left")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


messages3 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/home/admin/finetune-Qwen2-VL/M2_GB/M2_GB_36.png",
            },
            {"type": "text", "text": "What would you caption the character in this picture?"},
        ],
    }
]
# messages3 = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "video",
#                 "video": "test_data/1.mp4",
#                 "max_pixels": 360 * 420,
#                 "fps": 1.0,
#             },
#             {"type": "text", "text": "What kind of chess pieces in this picture?"},
#         ],
#     }
# ]

# messages = [messages3, messages12, messages13, messages14]
messages = [messages3]
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
pprint(output_text)
