## Task
You have been provided a persona-based chat dataset which consists of a hypothetical conversation between 2 personas A and B. Your task is to fine-tune any LLM on this dataset to inherit a persona and to respond in a humane way. 
[Link](https://huggingface.co/datasets/Cynaptics/persona-chat) to the dataset.

Here are a few points which you should be mindful of and will be considered in the judging criteria:
- Choice of the LLM.
- Quality and method of fine-tuning.
- Training loss/cost.
- Fine-tuned model performance on test samples.
- Compute constraint is to use only freely available options. (Eg: Google Colab, Kaggle, etc.)

## Submission Format
You are supposed to create a GitHub repository and add your scripts and images of your objective functions. A README file should be included which supports your approach with the following points:
- Choice of LLM.
- Choice of fine-tuning method.
- Justifications.
- Link to model weights uploaded in a public Hugging Face account.

Repository Structure:
```
Task-3/
├── scripts/
│   ├── script1.py
│   ├── script2.ipynb
├── images/
│   ├── image1.png
│   ├── image2.jpg
├── README.md
```

## Examples and References
Example of an inference script for a persona-based chat:
```
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


prompt = f"""
Person B has the following Persona information.

Persona of Person B: My name is David and I'm a 35 year old math teacher.
Persona of Person B: I like to hike and spend time in the nature.
Persona of Person B: I'm married with two kids.

Instruct: Person A and Person B are now having a conversation. 
Following the conversation below, write a response that Person B would say base on the
above Persona information. 
Please carefully consider the flow and context of the conversation below, and use the Person B's Persona information appropriately to generate a response that you think are 
the most appropriate replying for Person B.

Persona A: Morning! I think I saw you at the parent meeting, what's your name?

Output:
"""

# load base LLM model, LoRA params and tokenizer
model = AutoModelForCausalLM.from_pretrained("model-card", trust_remote_code=True)
model.to("cuda")

tokenizer = AutoTokenizer.from_pretrained("model-card", trust_remote_code=True)

# tokenize input prompt
input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()

# inference
with torch.inference_mode():
    outputs = model.generate(
        input_ids=input_ids, 
        max_new_tokens=50, 
        do_sample=True, 
        top_p=0.1,
        temperature=0.7
    )

# decode output tokens
outputs = outputs.detach().cpu().numpy()
outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
output = outputs[0][len(prompt):]
print(output)
```





