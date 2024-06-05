import dspy

from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot
from memorious import CoT
from rich import print

def init_stuff(model: str = "llama3"):
    lm = dspy.OllamaLocal(model=model, max_tokens=500)
    dspy.settings.configure(lm=lm)
    
    # Load math questions from the GSM8K dataset
    gsm8k = GSM8K()
    gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]
    
    print(f"[magenta]{len(gsm8k_devset)}[/magenta]")
    print(f"[magenta]{gsm8k_devset[0]}[/magenta]")
    
    
    print(f"[green]Set up the optimizer: we want to \"bootstrap\" (i.e., self-generate) 4-shot examples of our CoT program.[/green]")
    config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

    print(f"[green]Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.[/green]")
    teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
    optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset)
    

    print(f"[green]Set up the evaluator, which can be used multiple times.[/green]")
    evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)

    print(f"[green]Evaluate our `optimized_cot` program.[/green]")
    print(evaluate(optimized_cot))
    
    lm.inspect_history(n=10)
    
    print(f"Question: {gsm8k_devset[0]['question']}\nAnswer: {gsm8k_devset[0]['answer']}")
    
    print(f"{optimized_cot(gsm8k_devset[0]['question'])}")
    