from transformers import RobertaTokenizer, RobertaModel
import torch

# Load the tokenizer and model for CodeBERT
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")

print('CodeBERT embedding is:', model.config.hidden_size)

# Create a sample input to pass through CodeBERT
sample_code = "int main() { return 0; }"  # Example code snippet
inputs = tokenizer(sample_code, return_tensors="pt", padding=True, truncation=True)

# Pass the inputs through the CodeBERT model
with torch.no_grad():
    outputs = model(**inputs)

# Get the size of the embedding
embedding_size = outputs.last_hidden_state.size(-1)
print(f"The size of the CodeBERT embedding is: {embedding_size}")
