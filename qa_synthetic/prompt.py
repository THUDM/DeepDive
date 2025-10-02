GENERATE_QUESTION_KILT = r"""
You are given:

A path, which is an ordered sequence of entities (nodes) from a random walk on Wikipedia, each linked by specific relationships.
For each node, a description (node introduction) is provided, detailing key attributes such as names, places, events, or times.
Your task:

Identify one attribute from the last node in the path (this could be a name, place, event, or time), and construct a question whose answer is exactly the value of that attribute.
Your question must require the solver to follow the path step by step, using each node and relationship in order to arrive at the solution; the clues should be layered so that each step depends on the previous one.
Do not directly mention the actual names, places, dates, or events in the path or node introductions. Instead, paraphrase or generalize all clues and relationships, so the answer cannot be obtained by simple keyword matching. 

The question should include up to 15 unique clues, each clue building upon information from the previous node or relationship, reflecting the path's structure.
The question should be concise, logically connected, and solvable strictly using the information from the path and node descriptions. 
The relationship between the next node and the previous node is buried in the text of the previous node, and the connection between them must be mentioned and !
Every node's information should be mentioned in the question. 
The answer must be unique and must exactly match the value of the selected attribute from the last node. 
Try to keep the question length (number of characters) less than 800.
All time/name/event/place information in the puzzle needs to be obscured!


Input:
Paths: %(path)s 
Node Introductions: %(intro)s

Please strictly follow the output format below:

<question>: (A multi-step, path-ordered question in English)
<answer>: (The unique answer)
"""
