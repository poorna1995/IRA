

# Task Complexity Benchmark — Python dependencies

# Python >= 3.10 required

# Install: pip install -r requirements.txt

# 
# Core data

datasets==2.19.2
huggingface-hub==0.23.4
pandas==2.2.2
pyarrow==16.1.0
numpy==1.26.4

# Config & utilities

pyyaml==6.0.1
python-dotenv==1.0.1
tqdm==4.66.4
rich==13.7.1 # pretty terminal output

# Evaluation & metrics

scipy==1.13.1
scikit-learn==1.5.0

# LLM interfaces (for Block 2 estimators)

openai==1.35.3
anthropic==0.28.0
transformers==4.41.2 # for white-box entropy probing
torch==2.3.1 # CPU-only; add --extra-index-url for CUDA

# Optional: HF tokenizers (fast)

tokenizers==0.19.1

# Notebook

jupyter==1.0.0
ipykernel==6.29.5

# Testing

pytest==8.2.2
