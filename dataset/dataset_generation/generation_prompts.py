
PROMPT ={
"O" : """
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
    Output : "Given that 11 is a divisor of 66 and 66 is a divisor of 198, is it true that 11 is a divisor of 198?","Given that $11\mid66$ and $66\mid198$.$11\mid66$ means there exists an integer $m_1$ such that $66 = 11 \cdot m_1$ and $66\mid198$ means there exists an integer $m_2$ such that $198 = 66 \cdot m_2$.\n We need to show that $11\mid198$.\n From the first equation, we can express $66$ in terms of $11$ as $66 = 11 \cdot m_1$.\n Substitute this into the second equation:$$198 = 66 \cdot m_2 = (11 \cdot m_1) \cdot m_2$$ Now, let's simplify this:$$ 198 = 11 \cdot (m_1 \cdot m_2) $$ Since $m_1$ and $m_2$ are integers,   $ m_1 \cdot m_2$ is also an integer.\n Let's call it $m_3 = m_1 \cdot m_2$.\n Therefore, we can rewrite the equation as:$$198 = 11 \cdot m_3$$ This implies that $11\mid198$, as there exists an integer $m_3$ (specifically, $m_3 = m_1 \cdot m_2$) such that $198 = 11 \cdot m_3$. Hence, we've demonstrated mathematically that $11$ divides $198$.","orig","math","numbertheory","cns" """,
"R" : """
    You are an AI researcher working on a project to generate rephrased question and answer pairs. You are restricted to generate anything that is offensive or inappropriate. As an input, you will be given a question and answer pair. Your task is to generate a rephrased version of both the question and the answer. You can use any method you like, like replacing synonyms, reordering words, rearranging sentences, etc., so that it sounds like a new dataset altogether. The rephrased output should be grammatically correct and should preserve the meaning of the original sentence. Be as creative as you want. The question and answer pair will be given in the following format: "{question}","{answer}","orig". The rephrased sentence should be in the following format: "{rephrased_answer}","{rephrased_answer}","rephr". 
    The following examples will guide you how to generate rephrased sentences:

    Example 1:
    **Input:**
    "What is the capital of India?", "The capital of India is New Delhi.", "orig"

    **Output:**
    "Do you know the capital of India?","New Delhi is the capital of India.","rephr"

    Example 2:
    **Input:**
    "What is cryptography and do you know its importance?", "Cryptography is the practice of securing communication in the presence of adversaries, ensuring that only intended recipients can read messages and that data remains confidential, intact, and authentic. Its importance lies in its ability to protect privacy, maintain data integrity, authenticate identities, prevent repudiation, securely exchange keys, enable digital signatures, and safeguard against cyberattacks. Cryptography is crucial for national security, legal transactions, and the overall integrity of digital communications, making it indispensable in the digital age.", "orig"
    ", "orig"

    **Output:**
    "Can you tell me what cryptography is and state its importance", "Cryptography involves securing communication to prevent unauthorized access, ensuring messages are only readable by intended recipients and that data remains confidential, unaltered, and genuine.Cryptography's significance stems from its role in safeguarding privacy, ensuring data integrity, verifying identities, preventing denial of actions, facilitating secure key exchange, enabling digital signatures, and defending against cyber threats. It is essential for national security, legal processes, and the integrity of digital communications, highlighting its critical importance in the digital era.","rephr"
    """,
"V" : """
    You are a dataset augmentation specialist working on a project to generate various versions of a single question type using value modification. You are restricted to generate anything that is offensive or inappropriate. As an input, you will be given a question and answer pair and a new question similar to the given question. Your task is generate a new answer for the new question by using the same structure as in the question-answer pair. This will be mostly numerical or categorical value modification. You final goal is to form as many question- answer pair as possible by using the same structure. The question and answer pair will be given in the following format: "{question}","{answer}","orig" and "{new_question}". The output will be given in the following format: "{new_question}","{new_answer}","inpmod".

    The following examples will guide you how to generate INPUT MODIFICATION sentences:

    Example 1:
    **Input:**
    "What is 23+45?", "23+45 is 68.", "orig"
    "What is 23+46?"

    **Output:**
    "What is 23+46?", "23+46 is 69.", "inpmod"

    Example 2:
    **Input:**
    "There are three apples in a basket. If two more apples are added to the basket, how many apples will be there in the basket?", "If two more apples are added to the basket, there will be five apples in the basket.", "orig"

    "There are two ducks in a pond. If five more ducks are added to the pond, how many ducks will be there in the pond?"

    **Output:**
    "There are two ducks in a pond. If five more ducks are added to the pond, how many ducks will be there in the pond?","There are two ducks in a pond. If five more ducks are added to the pond, there will be seven ducks in the pond.", "inpmod"
    """,
"B" : """
    You are a dataset augmentation specialist working on a project to generate various versions of a single question type using backward reasoning. You are restricted to generate anything that is offensive or inappropriate. As an input, you will be given a question and answer pair and your task is to generate the same question using backward reasoning prompting. In backward reasoning, you have to mask the value of one of the numerical or categorical values in the question and answer pair and ask the user to find the value of the masked value, given the answer. So every backward question should have a masked value "x" and the last sentence of the question should be "What is the value of x?".
    You can be as creative as you want.
    The question and answer pair will be given in the following format: "{question}","{answer}","orig". The output will be given in the following format: "{new_question}","{new_answer}","bkwrd".

    The following examples will guide you how to generate BACKWARD REASONING sentences:

    Example 1:
    **Input:**
    Forward question-answer pair : "For $a = 5$ and $b = 7$, what is the greatest common divisor (gcd) and least common multiple (lcm) of $a$ and $b$?","To find the gcd of $5$ and $7$, we apply the Euclidean Algorithm: $7 = 5 \cdot 1 + 2$, $5 = 2 \cdot 2 + 1$, $2 = 1 \cdot 2 + 0$. So, the gcd of $5$ and $7$ is $1$. The lcm of $5$ and $7$ is given by $\frac{{|5 \cdot 7|}}{{\gcd(5, 7)}} = \frac{{35}}{{1}} = 35$.","orig"

    **Output:**

    "For a = 5 and b = y, the greatest common divisor(gcd) is 1 and the least common multiple (lcm) is 35. What is the value of y ?", "Answer: To solve for x given that the gcd of 5 and y is 1 and the lcm is 35, we can use the relationship between the gcd and lcm of two numbers. The formula for the lcm of two numbers a and b is given by |a x b| / gcd(a, b). \\ Given that the gcd of 5 and y is 1, we can substitute these values into the formula for the lcm: \\
    lcm(5, y) = (5 x y) / gcd(5, y) \\ Since the gcd is (1), the formula simplifies to: \\ lcm(5, y) = 5y \\ We are told that the lcm is 35, so we set 5y = 35 and solve for y:5y = 35 ; y = 35/5  ; y = 7 \\ Therefore, the value of y that satisfies the given conditions is 7.","bkwrd"

    Example 2:

    **Input:**
    Forward question-answer pair : "How is the division of 2044 by 8 performed, and what is the resulting quotient and remainder?","The division of 2044 by 8 is carried out by performing the long division of the positive numbers. When the obtained result of $2044 = 8 \cdot 255 + 4$ is multiplied by 1, it yields $2044 = 8 \cdot 255 + 4$, which leads to a positive remainder. Consequently, the quotient and remainder are $255$ and $4$, respectively.","orig"


    "If the division of x by 8 is performed, and the resulting quotient is 255 and the remainder is 4, What is the value of x", "To find the value of x, we can use the equation $x = 8 \cdot 255 + 4$. This equation represents the division process, where x is divided by 8 resulting in a quotient of 255 and a remainder of 4. Therefore, the value of x is calculated as $x = 8 \cdot 255 + 4$, which simplifies to $x = 2044$.", "bkwrd","math","numbertheory","cns"
    """
}

def get_prompt(prompt_type):
    return PROMPT[prompt_type]