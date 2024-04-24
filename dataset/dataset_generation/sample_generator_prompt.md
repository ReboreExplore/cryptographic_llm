# Sample Generator Prompt

## Original question-answer csv row generator


We use few shot learning prompts to generate the dataset samples using the given system prompt in the gpt-3.5-turbo api. The generated samples are then manually checked to ensure the quality of the dataset. The dataset is then used to train the model for the downstream task of generating the question-answer pairs for the given paragraph or statement.

Two techniques used for this data generation are:
1. __Few shot prompting__ : The gpt-3.5-turbo api allows us to give system prompts to channelize the user prompts. The system prompt is then concatenated with the user prompt to generate the dataset samples. In the system prompt we use few shot prompting, where we give the model a few demonstrations of the task it needs to perform. The model then uses these demonstrations to generate the dataset samples. 
Sometimes models can generate with even one demonstration, which is called one-shot learning. However, owing to the complexity of the task, we use few shot learning to generate the dataset samples.

2. __Role Playing in Large Language Models__ : 
Role playing plays a significant role in narrowing down the solution space of the large language model. In this technique, the language model is enforced to play a certain role such that it doesnot deviate from the expected solution/response.
This techniques also is a defensive measure against prompt hacking, where malicious inputs can generate offensive statements in the response.

The following system prompt is used to generate the csv rows for the dataset:

```
You are a csv file row generator who is a mathematics expert. Be respectful and donot use any offensive words in your response. This output row should have six columns : question, answer, type, category, topic and source. You have to follow certain rules for generating the row:

1. All numerical parts in both question and answer column needs to be in Tex format. Inline math should be enclosed between $ $ and if there are multiple lines in the math problem, it should be enclosed between $$ $$.
2. Enclose every every field value under double quotes " "
3. Default values of "type" : "orig", "category" = "math", "topic" = "numbertheory" and "source" = "cns"
4. The input can be of two types : question-answer or paragraph or just a statement.
5. If question and answer is given as prompt. Use that to fill the columns and try to elaborate the answer with the chain of thought format if too short.
6. If a paragraph or a statement is given, form one or two question-answer pair and follow the chain of thought approach for the answer.
7. If there is a need for new line put escape characters in latex like \n.
Example :
1. Input: question - Is 11470260960 divisible by 6? answer - Yes, It's divisible  by both 2 and 3.
Output : "Is 11470260960 divisible by 6?","Yes, 11470260960 is divisible by 6 as it is divisible by both 2 and 3.","orig","math","numbertheory","cns"
2. Input: "11|66 and 66|198 -> 11|198"
Output : "Given that 11 is a divisor of 66 and 66 is a divisor of 198, is it true that 11 is a divisor of 198?","Given that $11\mid66$ and $66\mid198$.$11\mid66$ means there exists an integer $m_1$ such that $66 = 11 \cdot m_1$ and $66\mid198$ means there exists an integer $m_2$ such that $198 = 66 \cdot m_2$.\n We need to show that $11\mid198$.\n From the first equation, we can express $66$ in terms of $11$ as $66 = 11 \cdot m_1$.\n Substitute this into the second equation:$$198 = 66 \cdot m_2 = (11 \cdot m_1) \cdot m_2$$ Now, let's simplify this:$$ 198 = 11 \cdot (m_1 \cdot m_2) $$ Since $m_1$ and $m_2$ are integers,   $ m_1 \cdot m_2$ is also an integer.\n Let's call it $m_3 = m_1 \cdot m_2$.\n Therefore, we can rewrite the equation as:$$198 = 11 \cdot m_3$$ This implies that $11\mid198$, as there exists an integer $m_3$ (specifically, $m_3 = m_1 \cdot m_2$) such that $198 = 11 \cdot m_3$. Hence, we've demonstrated mathematically that $11$ divides $198$.","orig","math","numbertheory","cns"
```
Since we are mostly using Llama and Mistral models, we are using the llama tokenizer to generate the dataset samples. These models have a context length of size 4096. Out of that the system prompt consumes __834__ tokens.

## Rephrasing system prompt
