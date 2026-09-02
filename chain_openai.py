from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# 1. Create a prompt template for the chat model
prompt_template = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the following question:\n\n{question}"
)

# 2. Model configuration
model = ChatOpenAI(model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) # temperature controls randomness in responses

# 3. Create a function to generate responses
def generate_response(question: str) -> str:

    # Manually format the prompt and invoke the model (commented out for now)
    # Format the prompt with the user's question
    # formatted_prompt = prompt_template.format_prompt(question=question)
    
    # Manually invoke the model with the formatted prompt (commented out for now)
    # Get the model's response
    # response = model.invoke(formatted_prompt.to_messages())
    
    # Manually parse the response content (commented out for now)
    # Parse and return the response as a string
    # return StrOutputParser().parse(response.content)

    # Chain the prompt template, model, and output parser together for a streamlined response generation
    chain = prompt_template | model | StrOutputParser()
    return chain.invoke({"question": question})


print("Welcome to the LangChain Chatbot! Type 'exit' to quit.")
# 4. Main loop to interact with the user
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    
    # Generate and print the response
    answer = generate_response(user_input)
    print(f"Bot: {answer}")

