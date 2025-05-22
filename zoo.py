import logging
import os
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import torch
from PIL import Image
import json

import fiftyone as fo
from fiftyone import Model, SamplesMixin
from fiftyone.core.labels import Classification, Classifications

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

DEFAULT_CLASSIFICATION_SYSTEM_PROMPT = """You specialize in comprehensive classification across medical domains including radiology images, histopathology patches, ophthalmology images, and dermatology images.

Unless specifically requested for single-class output, you may report multiple relevant classifications. Report all classifications as JSON array of predictions in the format: 

```json
{
    "classifications": [
        {
            "label": "descriptive medical condition or relevant label",
            "label": "descriptive medical condition or relevant label",
            ...,
        }
    ]
}
```
Always return your response as valid JSON wrapped in ```json blocks. 

You may report multiple lables if they are relevant. Do not report your confidence.
"""

DEFAULT_VQA_SYSTEM_PROMPT = """You are an expert across medical domains including radiology images, histopathology patches, ophthalmology images,
and dermatology images. You provide expert-level answers to medical questions.
"""


MEDGEMMA_OPERATIONS = {
    "vqa": {
        "system_prompt": DEFAULT_VQA_SYSTEM_PROMPT,
        "params": {},
    },
    "classify": {
        "system_prompt": DEFAULT_CLASSIFICATION_SYSTEM_PROMPT,
        "params": {},
    }
}

logger = logging.getLogger(__name__)

# Utility functions
def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class medgemma(SamplesMixin, Model):
    """A FiftyOne model for running Gemma3 vision tasks"""

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        prompt: str = None,
        system_prompt: str = None,
        quantized: bool = None,
        **kwargs
    ):

        self._fields = {}
        
        self.model_path = model_path
        self._custom_system_prompt = system_prompt  # Store custom system prompt if provided
        self._operation = operation
        self.prompt = prompt
        self.quantized = quantized
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        # Set dtype for CUDA devices
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else "auto"
        
        # Load model and processor
        logger.info(f"Loading model from {model_path}")

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": self.device,
            "torch_dtype":self.torch_dtype 
        }

        if self.quantized:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        logger.info("Loading processor")

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )

        self.model.eval()

    def _get_field(self):
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)

        return prompt_field

    @property
    def media_type(self):
        return "image"
    
    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value):
        if value not in MEDGEMMA_OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(MEDGEMMA_OPERATIONS.keys())}")
        self._operation = value

    @property
    def system_prompt(self):
        # Return custom system prompt if set, otherwise return default for current operation
        return self._custom_system_prompt if self._custom_system_prompt is not None else MEDGEMMA_OPERATIONS[self.operation]["system_prompt"]

    @system_prompt.setter
    def system_prompt(self, value):
        self._custom_system_prompt = value

    def _parse_json(self, s: str) -> Optional[Dict]:
        """Parse JSON from model output.
        
        Args:
            s: String output from the model to parse
            
        Returns:
            Dict: Parsed JSON dictionary if successful
            None: If parsing fails or input is invalid
            Original input: If input is not a string
        """
        if not isinstance(s, str):
            return s
            
        # Handle JSON wrapped in markdown code blocks
        if "```json" in s:
            try:
                s = s.split("```json")[1].split("```")[0].strip()
            except IndexError:
                logger.debug("Failed to extract JSON from markdown blocks")
                return None
        
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse error: {e}. First 200 chars: {s[:200]}")
            return None

    def _to_classifications(self, data: Dict) -> fo.Classifications:
        """Convert JSON classification data to FiftyOne Classifications.
        
        Args:
            data: Dictionary containing a 'classifications' list where each item has:
                - 'label': String class label
            
        Returns:
            fo.Classifications object containing the converted classification annotations
            
        Example input:
            {
                "classifications": [
                    {"label": "condition_1"},
                    {"label": "condition_2"}
                ]
            }
        """
        classifications = []
        
        try:
            # Extract the classifications list from the input dictionary
            classes = data.get("classifications", [])
            
            # Process each classification dictionary
            for cls in classes:
                try:
                    if not isinstance(cls, dict) or "label" not in cls:
                        logger.debug(f"Invalid classification format: {cls}")
                        continue
                        
                    classification = fo.Classification(
                        label=str(cls["label"]),
                    )
                    classifications.append(classification)

                except Exception as e:
                    logger.debug(f"Error processing classification {cls}: {e}")
                    continue

        except Exception as e:
            logger.debug(f"Error processing classifications data: {e}")
            
        return fo.Classifications(classifications=classifications)

    def _predict(self, image: Image.Image, sample=None) -> Union[fo.Classifications, str]:
        """Process a single image through the model and return predictions."""
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                self.prompt = str(field_value)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},               
                    {"type": "image", "image": image}  # Pass the PIL Image directly
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            ).to(self.device, dtype=self.torch_dtype)

        input_len = text["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **text, 
                max_new_tokens=8192, 
                do_sample=False
                )
            generation = generation[0][input_len:]

        output_text = self.processor.decode(generation, skip_special_tokens=True)

        # For VQA, return the raw text output
        if self.operation == "vqa":
            return output_text.strip()

        # For other operations, parse JSON and convert to appropriate format
        if self.operation == "classify":
            parsed_output = self._parse_json(output_text)
            return self._to_classifications(parsed_output)


    def predict(self, image, sample=None):
        """Process an image with the model.
        
        A convenience wrapper around _predict that handles numpy array inputs
        by converting them to PIL Images first.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)