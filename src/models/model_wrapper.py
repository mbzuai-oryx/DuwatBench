#!/usr/bin/env python3
"""
Model wrapper for evaluating various OCR and VLM models on DuwatBench
Implements ALL 13 models from Tables 2 & 3 of the paper

Based on paper: "DuwatBench: Bridging Language and Visual Heritage"
Table 2: Full Image Evaluation
Table 3: With Bounding Boxes Evaluation
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Union
from PIL import Image
import base64
from io import BytesIO
import numpy as np

# Import API keys from config
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config.eval_config import GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, HUGGINGFACE_TOKEN
except ImportError:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")


class BaseModel(ABC):
    """Base class for all OCR/VLM models"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def transcribe(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """
        Transcribe Arabic text from image

        Args:
            image: PIL Image
            prompt: Optional prompt for VLMs

        Returns:
            Transcribed Arabic text
        """
        pass

    def encode_image_base64(self, image: Image.Image, max_size_mb: float = 5.0) -> str:
        """Encode PIL Image to base64 string with size limit and proper mode handling

        Note: Claude API measures base64 string size, not PNG size.
        Base64 encoding adds ~33% overhead (4/3 ratio).
        """
        # Convert problematic image modes to RGB
        # P = palette mode, CMYK = print colors, LA = grayscale with alpha, etc.
        if image.mode in ('CMYK', 'P', 'LA', 'PA', 'I', 'F'):
            image = image.convert('RGB')
        elif image.mode == 'RGBA':
            # Keep RGBA for transparency support in PNG
            pass
        elif image.mode != 'RGB':
            # Any other mode, convert to RGB
            image = image.convert('RGB')

        # Create a copy to avoid modifying the original
        img_to_encode = image.copy()

        # Convert MB limit to bytes for base64 comparison
        max_size_bytes = int(max_size_mb * 1024 * 1024)

        # Iteratively resize until under size limit
        max_iterations = 10  # Increased from 5 to handle very large images
        for iteration in range(max_iterations):
            buffered = BytesIO()
            img_to_encode.save(buffered, format="PNG", optimize=True)

            # Encode to base64 and check the actual base64 size (what Claude sees)
            b64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            b64_size_bytes = len(b64_str.encode('utf-8'))

            if b64_size_bytes <= max_size_bytes:
                # Size is acceptable
                return b64_str

            # Need to resize - reduce by 30% each iteration (more aggressive)
            scale = 0.7
            new_width = int(img_to_encode.width * scale)
            new_height = int(img_to_encode.height * scale)
            img_to_encode = img_to_encode.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # If still too large after max iterations, return the smallest version
        buffered = BytesIO()
        img_to_encode.save(buffered, format="PNG", optimize=True)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


# ============================================================================
# OPEN-SOURCE MODELS (8 models)
# ============================================================================

class LlavaModel(BaseModel):
    """
    Llava-v1.6-mistral-7b-hf
    From Table 2: CER=1.0959, WER=1.7872, chrF=0.4810
    """

    def __init__(self):
        super().__init__("Llava-v1.6-mistral-7b-hf")
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            import torch

            self.processor = LlavaNextProcessor.from_pretrained(
                "llava-hf/llava-v1.6-mistral-7b-hf"
            )
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                "llava-hf/llava-v1.6-mistral-7b-hf",
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except ImportError:
            raise ImportError("Please install transformers and torch: pip install transformers torch")

    def transcribe(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        # Use passed prompt or default Arabic prompt
        if prompt is None:
            prompt = "اقرأ النص العربي في الصورة وانسخه بالضبط. أخرج النص العربي فقط."

        # LLaVA-Next uses a conversation format with the image token
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=200)
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)

        # Extract assistant response
        if "ASSISTANT:" in generated_text:
            generated_text = generated_text.split("ASSISTANT:")[-1].strip()

        return generated_text


class EasyOCRModel(BaseModel):
    """
    EasyOCR
    From Table 2: CER=0.7857, WER=1.0213, chrF=7.7420
    """

    def __init__(self):
        super().__init__("EasyOCR")
        try:
            import easyocr
            self.reader = easyocr.Reader(['ar'], gpu=True)
        except ImportError:
            raise ImportError("Please install easyocr: pip install easyocr")

    def transcribe(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        import numpy as np
        img_array = np.array(image)
        results = self.reader.readtext(img_array, detail=0)
        return ' '.join(results)


class InternVL3Model(BaseModel):
    """
    InternVL3-8B
    From Table 2: CER=0.7461, WER=0.8781, chrF=10.3251
    """

    def __init__(self):
        super().__init__("InternVL3-8B")
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            import torchvision.transforms as T
            from PIL import Image as PILImage

            model_path = "OpenGVLab/InternVL3-8B"
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map='auto'
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
            self.torch = torch

            # Build image transform following InternVL3 preprocessing
            self.image_transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        except ImportError as e:
            raise ImportError(f"Please install required packages: pip install transformers torch einops timm torchvision. Error: {e}")

    def transcribe(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        # Use passed prompt or default Arabic prompt
        if prompt is None:
            prompt = "<image>\nاقرأ النص العربي في الصورة وانسخه بالضبط. أخرج النص العربي فقط."
        elif not prompt.startswith("<image>"):
            # Ensure prompt has image token for InternVL3
            prompt = f"<image>\n{prompt}"

        # Convert image to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Preprocess image using standard InternVL3 preprocessing
        pixel_values = self.image_transform(image).unsqueeze(0)
        pixel_values = pixel_values.to(self.torch.bfloat16).to(self.model.device)

        # Generate response
        generation_config = dict(max_new_tokens=200, do_sample=False)
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config
        )

        return response.strip()


class QwenVLModel(BaseModel):
    """
    Qwen2.5-VL-7B and Qwen2.5-VL-72B-Instruct
    From Table 2:
    - 7B: CER=0.6499, WER=0.8080, chrF=19.1702
    - 72B: CER=0.6969, WER=0.8587, chrF=29.2649
    """

    def __init__(self, model_size="7B"):
        model_name = f"Qwen2.5-VL-{model_size}-Instruct"
        super().__init__(model_name)

        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            import torch

            model_path = f"Qwen/{model_name}"
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch")

    def transcribe(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        # Use passed prompt or default Arabic prompt
        if prompt is None:
            prompt = "اقرأ النص العربي في الصورة وانسخه بالضبط. أخرج النص العربي فقط."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=200)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Extract response after assistant tag
        if "assistant\n" in generated_text:
            generated_text = generated_text.split("assistant\n")[-1]

        return generated_text.strip()


class Gemma3Model(BaseModel):
    """
    gemma-3-27b-it
    From Table 2: CER=0.6374, WER=0.7683, chrF=38.8290 (Best open-source)
    """

    def __init__(self):
        super().__init__("gemma-3-27b-it")
        try:
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration
            import torch

            # Gemma 3 27B vision model
            model_path = "google/gemma-3-27b-it"
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            ).eval()
            self.torch = torch
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch")

    def transcribe(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        # Use passed prompt or default Arabic prompt
        if prompt is None:
            prompt = "اقرأ النص العربي في الصورة وانسخه بالضبط. أخرج النص العربي فقط."

        # Convert image to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Format message with image using chat template format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Apply chat template and process
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=self.torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        # Generate response
        with self.torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
            generation = generation[0][input_len:]

        result = self.processor.decode(generation, skip_special_tokens=True)

        return result.strip()


class TrOCRModel(BaseModel):
    """
    RayR1/trocr-base-arabic-handwritten (Arabic-specific OCR model)
    From Table 2: CER=1.0343, WER=1.0440, chrF=0.7578

    Note: TrOCR is a dedicated OCR model that doesn't use prompts.
    It directly extracts text from images.
    """

    def __init__(self):
        super().__init__("trocr-base-arabic-handwritten")
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch

            model_path = "RayR1/trocr-base-arabic-handwritten"
            self.processor = TrOCRProcessor.from_pretrained(model_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path)

            # Try GPU, fallback to CPU if out of memory
            self.device = "cpu"
            if torch.cuda.is_available():
                try:
                    # Check if there's enough GPU memory (need ~500MB free)
                    free_memory = torch.cuda.mem_get_info()[0] / (1024**3)  # GB
                    if free_memory > 1.0:
                        self.model = self.model.to("cuda")
                        self.device = "cuda"
                        print(f"TrOCR loaded on GPU ({free_memory:.1f} GB free)")
                    else:
                        print(f"GPU memory low ({free_memory:.2f} GB free), using CPU")
                except Exception as e:
                    print(f"GPU not available ({e}), using CPU")

        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch")

    def transcribe(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        # TrOCR doesn't use prompts - it's a dedicated OCR model
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values, max_new_tokens=200)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()


class MBZUAIAINModel(BaseModel):
    """
    MBZUAI/AIN (Arabic-specific model)
    From Table 2: CER=0.6688, WER=0.8192, chrF=22.0817
    """

    def __init__(self):
        super().__init__("MBZUAI/AIN")
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            import torch

            model_path = "MBZUAI/AIN"
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch")

    def transcribe(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        # Use passed prompt or default Arabic prompt
        if prompt is None:
            prompt = "اقرأ النص العربي في الصورة وانسخه بالضبط. أخرج النص العربي فقط."

        # MBZUAI/AIN is based on Qwen2-VL, so it uses the same conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process the conversation
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=200)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Extract response after assistant tag
        if "assistant\n" in generated_text:
            generated_text = generated_text.split("assistant\n")[-1]

        return generated_text.strip()


# ============================================================================
# CLOSED-SOURCE MODELS (5 models)
# ============================================================================

class ClaudeModel(BaseModel):
    """
    claude-sonnet-4.5
    From Table 2: CER=1.1806, WER=1.0795, chrF=27.6271
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("claude-sonnet-4.5")
        self.api_key = api_key or ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Please set it in config/api_keys.py or as environment variable")

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            # Use the exact model ID from paper
            self.model_id = "claude-sonnet-4-20250514"
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

    def transcribe(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        # Use passed prompt or default Arabic prompt
        if prompt is None:
            prompt = "اقرأ النص العربي في الصورة وانسخه بالضبط. أخرج النص العربي فقط."

        base64_image = self.encode_image_base64(image)

        message = self.client.messages.create(
            model=self.model_id,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        return message.content[0].text.strip()


class GeminiModel(BaseModel):
    """
    gemini-1.5-flash and gemini-2.5-flash
    From Table 2:
    - 1.5-flash: CER=0.9117, WER=1.0258, chrF=41.9275
    - 2.5-flash: CER=0.3160, WER=0.4164, chrF=59.9590 (BEST OVERALL)
    """

    def __init__(self, model_version="2.5-flash", api_key: Optional[str] = None):
        model_name = f"gemini-{model_version}"
        super().__init__(model_name)
        self.api_key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Please set it in config/api_keys.py or as environment variable")

        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            # Model names need 'models/' prefix for the API
            if model_version == "1.5-flash":
                model_version = "1.5-flash"
                model_name = f"gemini-{model_version}"
            self.model_id = f"models/{model_name}"
        except ImportError:
            raise ImportError("Please install google-genai: pip install google-genai")

    def transcribe(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        # Use passed prompt or default Arabic prompt
        if prompt is None:
            prompt = "اقرأ النص العربي في الصورة وانسخه بالضبط. أخرج النص العربي فقط."

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[prompt, image]
        )

        return response.text.strip()


class OpenAIModel(BaseModel):
    """
    gpt-4o and gpt-4o-mini
    From Table 2:
    - gpt-4o-mini: CER=0.5330, WER=0.6826, chrF=27.7021
    - gpt-4o: CER=0.8296, WER=0.9797, chrF=17.1246
    """

    def __init__(self, model_version="gpt-4o-mini", api_key: Optional[str] = None):
        super().__init__(model_version)
        self.api_key = api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set. Please set it in config/api_keys.py or as environment variable")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.model_id = model_version
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def transcribe(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        # Use passed prompt or default Arabic prompt
        if prompt is None:
            prompt = "اقرأ النص العربي في الصورة وانسخه بالضبط. أخرج النص العربي فقط."

        base64_image = self.encode_image_base64(image)

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        return response.choices[0].message.content.strip()


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_model(model_name: str, **kwargs) -> BaseModel:
    """
    Factory function to create model instances

    ALL 13 models from Tables 2 & 3:

    Open-source (8):
    1. Llava-v1.6-mistral-7b-hf
    2. EasyOCR
    3. InternVL3-8B
    4. Qwen2.5-VL-7B
    5. Qwen2.5-VL-72B-Instruct
    6. gemma-3-27b-it
    7. trocr-base-arabic-handwritten*
    8. MBZUAI/AIN*

    Closed-source (5):
    9. claude-sonnet-4.5
    10. gemini-1.5-flash
    11. gemini-2.5-flash
    12. gpt-4o-mini
    13. gpt-4o

    Args:
        model_name: Name of the model
        **kwargs: Additional arguments for model initialization

    Returns:
        BaseModel instance
    """
    model_map = {
        # Open-source
        "Llava-v1.6-mistral-7b-hf": lambda: LlavaModel(),
        "EasyOCR": lambda: EasyOCRModel(),
        "InternVL3-8B": lambda: InternVL3Model(),
        "Qwen2.5-VL-7B": lambda: QwenVLModel("7B"),
        "Qwen2.5-VL-72B-Instruct": lambda: QwenVLModel("72B"),
        "gemma-3-27b-it": lambda: Gemma3Model(),
        "trocr-base-arabic-handwritten": lambda: TrOCRModel(),  # RayR1/trocr-base-arabic-handwritten
        "MBZUAI/AIN": lambda: MBZUAIAINModel(),

        # Closed-source
        "claude-sonnet-4.5": lambda: ClaudeModel(**kwargs),
        "gemini-1.5-flash": lambda: GeminiModel("1.5-flash", **kwargs),
        "gemini-2.5-flash": lambda: GeminiModel("2.5-flash", **kwargs),
        "gpt-4o-mini": lambda: OpenAIModel("gpt-4o-mini", **kwargs),
        "gpt-4o": lambda: OpenAIModel("gpt-4o", **kwargs),
    }

    if model_name not in model_map:
        available = "\n  ".join(model_map.keys())
        raise ValueError(
            f"Model '{model_name}' not supported.\n\n"
            f"Available models (13 total from paper Tables 2 & 3):\n  {available}"
        )

    return model_map[model_name]()


def list_all_models():
    """Returns list of all 13 models from the paper"""
    return [
        # Open-source (8)
        "Llava-v1.6-mistral-7b-hf",
        "EasyOCR",
        "InternVL3-8B",
        "Qwen2.5-VL-7B",
        "Qwen2.5-VL-72B-Instruct",
        "gemma-3-27b-it",
        "trocr-base-arabic-handwritten",
        "MBZUAI/AIN",
        # Closed-source (5)
        "claude-sonnet-4.5",
        "gemini-1.5-flash",
        "gemini-2.5-flash",
        "gpt-4o-mini",
        "gpt-4o"
    ]
