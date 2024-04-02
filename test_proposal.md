\section*{Motivation}
Large Language models (LLM), for e.g. GPT-4, Falcon, LLaMA have made a remarkable impact in the past few years in solving downstream Natural Language Processing (NLP) tasks like text comprehension and generation. However, these models have shown strange accuracy patterns on mathematical tasks involving multi-step reasoning ( \cite{minerva-21},\cite{math-21} ). Solving these problems involves understanding natural language input, recalling relevant world knowledge, performing mathematical computations, and generating precise mathematical expressions. Thus hallucination even in one step can propagate to the next step and lead to a wrong answer.
Therefore, quantitative reasoning problems serve as a valuable area for testing the capabilities of language models and as a testbed for developing robust quantitative reasoning solvers that can assist humans in scientific and technical fields and enhance models' versatility.


\section*{Problem Statement}
Mathematical problems require a combination of natural language understanding, logical thinking, and computational skills. Large language models are probabilistic models that are trained on large corpora of text from the internet. They operate by predicting the most probable next word or token based on the input context, influenced by factors like temperature and internal model parameters. While these models are proficient in a wide range of language tasks, there is room for improvement in their mathematical problem-solving abilities. Research by \cite{codex-21} and \cite{mathqa-python-21} have shown that domain-specific datasets on mathematical problems and training can improve the performance of large language models on mathematical problems.

Furthermore, it is essential to note that fine-tuning the foundational models of large language models for specific tasks, like mathematical problem-solving, offers a cost-effective alternative to adopting larger models such as GPT-4. The fine-tuning process allows for customization to the specific requirements of the task, which can yield highly competitive results without the substantial expenses associated with training and deploying larger models.

Therefore, in this project I intend to focus on the specialized mathematical domain of cryptography and develop fine tune a foundational large language model to solve mathematical problems involving cryptography. The model will be trained on a dataset of mathematical problems involving cryptography and will be fine-tuned on the same dataset. A comparative analysis of the performance of the model will be done with the baseline models like GPT-3 and LLaMA. The results will be reported in terms of accuracy and perplexity.


 model that can solve mathematical problems involving cryptography. The model will be trained on a dataset of mathematical problems involving cryptography and will be fine-tuned on the same dataset. The model will be evaluated on a test set of mathematical problems involving cryptography. The model will be evaluated on the basis of accuracy and perplexity. The model will be compared with the baseline model of GPT-2 and the results will be reported.


 Fine-tuning the foundational models of large language models for specific tasks, like mathematical problem-solving, offers a cost-effective alternative to adopting larger models such as GPT-4. The fine-tuning process allows for customization to the specific requirements of the task, which can yield highly competitive results without the substantial expenses associated with training and deploying larger models.


Touvron et al.\cite{LLaMA2023} trained a series of language models called LLaMA \cite{LLaMA2023} with billions of parameters (from 7B to 65B parameters) and publicly available datasets, which outperformed several then-state-of-the-art LLMs. LLaMA models are able to achieve comparable performance with smaller model sizes, as compared to the counterpart GPT-3\cite{gpt3-20}, performing exceptionally well, especially on question answering and code generation. Around the same time, Yang et al. \cite{yang2023gpt} also conducted research where they showed close to perfect accuracy with multi-digit arithmetic problems (eight+ digits) with their fine-tuned model MathGLM. MathGLM is a 2 billion parameter model, trained on the GLM-Dialog model \cite{zhang2023glmdialog}.



\section*{Goals and Methodology}
\subsection*{Mandatory Goals}
\begin{itemize}
    \item  \textbf{Dataset Preparation}: Preparing a cryptography-based dataset for the mathematical field of \textit{cryptography} (crypto-dataset) to cover the \textit{Introduction to Number Theory} chapter of the book \textit{Cryptography and Network Security - Principles and Practice} by Willian Stallings and some modern basic cryptography concepts from the book \textit{Introduction to Modern Cryptography} by Jonathan Katz and Yehuda Lindell. Additional resources from the internet will also be used to supplement the dataset. The dataset will be prepared in a format that can be used for fine-tuning the foundational llm choosed. 

    This step also includes exploratory research on an optimum data representation format for representing each sample in the dataset. Math datasets GHOSTS, MATH-QA and GSM8K will be studied for this purpose. Additional information about the sample like the difficulty level and domain will also be added to the json object of the sample for the model to learn more about the sample. Also, the choice between multistep solution or multiple solution or a single solution will be made for best optimization of the model.  

    This is the most important step of the project and will we partially done manually and partially using the LangChain framework to get the best results in natural language.
    
    \item \textbf{Finetuning the LLM}: The performance of an llm can be enhanced with either fine-tuning or prompt engineering. In this project, fine-tuning will be used to enhance the performance of the llm.
    Configuring various hyperparameters like the choice of model architecture, layers, size, etc with the custom dataset generated and fine-tuning the pre-trained LLMs. Parameter efficient fine tuning methods like LoRA and Qora will be used for this purpose, instead of full parameter fine tuning. 
     The results will be based on evaluation metrics such as accuracy and precision in the numerical results obtained.

    The LangChain framework is intended to be used for this implementation.

    \item \textbf{Comparative Analysis}: Various language models will be tested against the "crypto-dataset" and the mathematical capabilities of each will be evaluated.
\end{itemize}

\subsection*{Optional Goals}
\begin{itemize}
    \item \textbf{Prompt Engineering:} Further analysis of various prompt engineering techniques can be implemented to compare the performance of the model with the fine tuning technique proposed in the project.
    \item \textbf{User Study}: A user study can be conducted with a graduate or undergraduate student specializing in cryptography and comparing the accuracies of both the human subjects and the AI model.
\end{itemize}