model=teknium/OpenHermes-2.5-Mistral-7B
model="llama3"
model="nvidia/Llama3-ChatQA-1.5-8B"
model="google/gemma-7b"
model="microsoft/Phi-3-small-8k-instruct"
model="meta-llama/Llama-2-7b-hf" # set to the specific Hugging Face model ID you wish to use.
model="microsoft/Phi-3-mini-4k-instruct"
model="microsoft/Phi-3-small-128k-instruct"
model="google/gemma-7b-it"
model="meta-llama/Meta-Llama-3-8B"
model="meta-llama/Meta-Llama-3-8B-Instruct"
model="google/gemma-7b-it"
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
model="Groq/Llama-3-Groq-8B-Tool-Use"

num_shard=1 # set to the number of shards you wish to use.
volume=$HOME/data # share a volume with the Docker container to avoid downloading weights every run

# --max-batch-prefill-tokens=131122 --max-total-tokens=131072 --max-input-tokens=131071
docker run --gpus all \
	--shm-size 2g \
       	-p 8081:80 \
       	-v $volume:/data \
	-e HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} \
	--model-id $model \
	--num-shard $num_shard \
	--max-batch-prefill-tokens=2122 \
	--max-total-tokens=2072 \
	--max-input-tokens=2071 

