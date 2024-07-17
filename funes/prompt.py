from langchain_core.prompts import PromptTemplate

basic_react_prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked. Do Thought one step at a time.
Use Action to run only one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()


basic_react_prompt_template = """
You are a smart research assistant. Think step by step what would be \
the steps to follow in a plan and then use the tools to look up information you need, \
for example about the weight of different dogs or combinations of them based on their breed. \
There is also a calculator to calculate mathematical expressions like "(2 * 3) + 1" . \
You are allowed to make multiple calls to the tools to collect Facts. \
If you need to look up some information before asking a follow up question, \
you are allowed to do that by means of the tools! 

    
Facts:

{facts}

Only look up information when you are sure of what you want. \
If the answer is in the facts, do no call any tool and just return the fact containing the answer.
""".strip()

basic_template = PromptTemplate(template=basic_react_prompt_template, input_variables=["facts"])


