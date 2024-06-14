import dspy

from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.datasets.hotpotqa import HotPotQA
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch
from memorious import CoT
from rich import print

# Set up llama3 with a VLLM client, served on four GPUs. Please note that these URLs will not work for you; you'd need to refer to the documentation to set up your own VLLM/SGLANG server(s).
model="llama3"
model="nvidia/Llama3-ChatQA-1.5-8B"
model="google/gemma-7b"
model="microsoft/Phi-3-small-8k-instruct"
model="meta-llama/Llama-2-7b-hf"
lm = dspy.HFClientVLLM(model=model, port=8081, url="http://localhost")
# lm = dspy.OllamaLocal(model=model, max_tokens=500)
# dspy.settings.configure(lm=lm)
dspy.configure(lm=lm)
qa = dspy.ChainOfThought('question -> answer')

print("Sending question...")
response = qa(question="What is the capital of Paris?") #Prompted to vllm_llama2
print(response.answer)

# example_qa = dspy.Example(question="Which is the region whose capital is Zaragoza?", answer= "Aragon") #.with_inputs("question")

# # input_key_only = article_summary.inputs()
# # non_input_key_only = article_summary.labels()
# example_qa

