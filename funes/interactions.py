import dspy


class FactualSortQA(dspy.Signature):
    """Answer questions with short-plain factual answers. Remove quotations (e.g. King Alfred is preferred over "King Alfred") when not needed."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Use only between 1 and 5 words")
