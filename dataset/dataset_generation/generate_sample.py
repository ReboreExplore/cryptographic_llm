# Import libraries
import json
import openai
from IPython.display import display, Markdown
from generation_prompts import get_prompt


USER_PROMPT = """
    The ciphertext-only attack is the easiest to defend against because the opponent has the least amount of information to work with. In many cases, however, the analyst has more information. The analyst may be able to capture one or more plaintext messages as well as their encryptions. Or the analyst may know that certain plaintext patterns will appear in a message. For example, a file that is encoded in the Postscript format always begins with the same pattern, or there may be a standardized header or banner to an electronic funds transfer message, and so on. All these are examples of known plaintext. With this knowledge, the analyst may be able to deduce the key on the basis of the way in which the known plaintext is transformed. The known-plaintext attack is the most serious of the attacks because it is the easiest to mount and because it is the most difficult to defend against.
    """
# Get API key from config.json in the same directory
# You are required to create a config.json file with your OpenAI API key in the root directory.
with open('config.json') as f:
   data = json.load(f)
   api_key = data['OPENAI_API_KEY']

# Set up your OpenAI API key
openai.api_key = api_key

# Define function for printing long strings as markdown
md_print = lambda text: display(Markdown(text))

# Call ChatGPT API with prompt
def call_GPT(prompt):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}])
    response = completion.choices[0].message.content
    md_print(f'Me: {prompt}')
    md_print(f'LLM_Output: {response}')


if __name__ == "__main__":

    # Get the prompt from the user
    print("Choose the prompt you want to generate:\n")
    print("O - Original prompt\n")
    print("R - Rephrased prompt\n")
    print("V - Value modification\n")
    print("B - Backward prompt\n")

    input_choice = input("Enter your choice (O/R/V/B): ")
    if input_choice == 'O' or input_choice == 'o':
       prompt = get_prompt('O')
    elif input_choice == 'R' or input_choice == 'r':
        prompt = get_prompt('R')   
    elif input_choice == 'V' or input_choice == 'v':
        prompt = get_prompt('V')
    elif input_choice == 'B' or input_choice == 'b':
        prompt = get_prompt('B')
    else:
        print("Invalid choice. Please enter a valid choice.")
        exit()
    call_GPT(prompt + USER_PROMPT)