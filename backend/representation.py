### TODO: USE LLAMA


SYSTEM_PROMPT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful, respectful and honest assistant for labeling topics.
<|eot_id|>
"""

EXAMPLE_PROMPT = """
<|start_header_id|>user<|end_header_id|>
I have a topic that contains the following documents:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.


The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.


Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.<|eot_id|>


<|start_header_id|>assistant<|end_header_id|>
Environmental impacts of eating meat<|eot_id|>
"""

MAIN_PROMPT = """
<|start_header_id|>user<|end_header_id|>
I have a topic that contains the following documents:
[DOCUMENTS]


The topic is described by the following keywords: '[KEYWORDS]'.


Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.<|eot_id|>


<|start_header_id|>assistant<|end_header_id|>
"""

PROMPT_TEMPLATE = SYSTEM_PROMPT + EXAMPLE_PROMPT + MAIN_PROMPT 

