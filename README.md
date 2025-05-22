# Implementing MedGemma as a Remote Zoo Model for FiftyOne

This repository integrates Google's MedGemma models with FiftyOne, allowing you to easily use these powerful medical AI models for analyzing and classifying medical images in your FiftyOne datasets.

## What is MedGemma?

MedGemma is a collection of [Gemma 3](hhttps://huggingface.co/collections/google/medgemma-release-680aade845f90bec6a3f60c4) variants that are trained specifically for medical text and image comprehension. These models excel at understanding various medical imaging modalities including:

- Chest X-rays
- Dermatology images
- Ophthalmology images
- Histopathology slides

This integration is for **MedGemma 4B**, a multimodal version that can process both images and text

## Installation

First, ensure you have FiftyOne installed:

```bash
pip install fiftyone
```

Then register this repository as a custom model source:

```python
import fiftyone.zoo as foz
foz.register_zoo_model_source("https://github.com/harpreetsahota204/medgemma", overwrite=True)
```

## Usage

### Download and Load the Model

```python
import fiftyone.zoo as foz

# Download the model (only needed once)
foz.download_zoo_model(
    "https://github.com/harpreetsahota204/medgemma",
    model_name="google/medgemma-4b-it", 
)

# Load the model
model = foz.load_zoo_model(
    "google/medgemma-4b-it"
)
```

### Setting the Operation Mode

The model supports two main operations:

```python
# For medical image classification
model.operation = "classify"

# For visual question answering on medical images
model.operation = "vqa"
```

### Classification Example

```python
import fiftyone as fo


from fiftyone.utils.huggingface import load_from_hub

dataset = load_from_hub(
    "Voxel51/MedXpertQA",
    name="MedXpertQA",
    max_samples=10,
    overwrite=True
    )

# Set classification parameters
model.operation = "classify"
model.prompt = "What medical conditions are visible in this image?"

# Run classification on the dataset
dataset.apply_model(model, label_field="medgemma_classifications")

```

### Visual Question Answering Example

```python
# Set VQA parameters
model.operation = "vqa"
model.prompt = "Is there evidence of pneumonia in this chest X-ray? Explain your reasoning."

# Apply to dataset
dataset.apply_model(model, label_field="pneumonia_assessment")

```

### Custom System Prompts

You can customize the system prompt to better suit your specific needs:

```python
model.system_prompt = """You are an expert dermatologist specializing in skin cancer detection.
Analyze the provided skin lesion image and determine if there are signs of malignancy.
Provide your assessment in JSON format with detailed observations."""
```

## Performance Considerations

- For optimal performance, a CUDA-capable GPU is recommended

- The model supports quantization to reduce memory requirements:
  ```python
  model = foz.load_zoo_model(
      "google/medgemma-4b-it",
      quantized=True  # Enable 4-bit quantization
  )
  ```

# 👩🏽‍💻 Example notebook

You can refer to the [example notebook](using_medgemma_zoo_model.ipynb) to get hands on.


## License

MedGemma is governed by the [Health AI Developer Foundations terms of use](https://developers.google.com/health-ai-developer-foundations/terms).

This integration is licensed under the Apache 2.0 License.

## Notes

- This integration is designed for research and development purposes
- Always validate model outputs in clinical contexts
- Review the [MedGemma documentation](https://developers.google.com/health-ai-developer-foundations/medgemma) for detailed information about the model's capabilities and limitations
