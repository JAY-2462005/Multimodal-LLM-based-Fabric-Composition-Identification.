# Multimodal Fabric Composition Identification

This repository contains experimental multimodal workflows for estimating fabric composition(in %) from textile images using large vision-language models.

The project includes two main notebook implementations:

- `python_scripts/llama.ipynb`: Fine-tuning and evaluation of `meta-llama/Llama-3.2-11B-Vision-Instruct` with LoRA adapters and a regression head for predicting fiber percentage composition.
- `python_scripts/blip.ipynb`: Training and evaluation using a BLIP-2 based architecture (`Salesforce/blip2-opt-2.7b`) for the same fabric composition regression task.

## Repository Structure

- `python_scripts/`
  - `llama.ipynb`: LLaMA-based vision regression workflow.
  - `blip.ipynb`: BLIP-2 based regression workflow.
- `sample_predictions/`: Example output figures and visualization artifacts.
- `Research_paper/`: Supporting paper or publication material.
- `LICENSE`: MIT license file.

## Dataset Expectations

These notebooks assume the following files and folders exist at the repository root:

- `train.csv`, `val.csv`, `test.csv`
- `train/`, `val/`, `test/` image folders

The CSV files should contain `image_id` values and fiber composition columns such as:

- `cotton`
- `wool`
- `polyester`
- `linen`
- `silk`
- `other_fibres`

## Requirements

The notebooks rely on the following Python libraries:

- `transformers`
- `datasets`
- `accelerate`
- `peft`
- `bitsandbytes`
- `torch`
- `torchvision`
- `pandas`
- `Pillow`
- `scikit-learn`
- `numpy`
- `huggingface_hub`

Install the core dependencies with:

```bash
pip install -q -U transformers datasets accelerate peft bitsandbytes torch torchvision pandas pillow scikit-learn numpy huggingface_hub
```

## Usage

1. Open the desired notebook in `python_scripts/`.
2. Configure the Hugging Face authentication token in the notebook before loading or saving models.
3. Verify the dataset CSV files and image folders are present in the repository root.
4. Run the notebook cells sequentially.

## Notes

- The notebooks are designed for research and experimentation rather than production deployment.
- Large model weights and quantized checkpoints may require significant GPU memory.
- The repository does not include the raw dataset or pre-trained model files.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
