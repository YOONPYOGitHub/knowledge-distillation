"""Prompt \ud15c\ud50c\ub9bf (\ub4dc\ub86d\ub2e4\uc6b4\uc5d0\uc11c \uc120\ud0dd)"""

PROMPT_TEMPLATES: dict[str, str] = {
    "(직접 입력)": "",
    "Completion": "The history of artificial intelligence began",
    "Factual": "The capital of France is",
    "Narrative": "Once upon a time in a small village,",
    "Technical": "In machine learning, gradient descent is",
    "Dialogue": "User: What is the meaning of life?\nAssistant:",
    "WikiText style": " = Valkyria Chronicles III = \n\n Senjō no Valkyria 3 ",
}


PARAM_PRESETS: dict[str, dict] = {
    "Balanced (T=0.8)": {
        "temperature": 0.8, "top_p": 0.9, "top_k": 50, "do_sample": True,
    },
    "Conservative (T=0.3)": {
        "temperature": 0.3, "top_p": 0.8, "top_k": 40, "do_sample": True,
    },
    "Creative (T=1.2)": {
        "temperature": 1.2, "top_p": 0.95, "top_k": 80, "do_sample": True,
    },
    "Greedy": {
        "temperature": 1.0, "top_p": 1.0, "top_k": 0, "do_sample": False,
    },
}
