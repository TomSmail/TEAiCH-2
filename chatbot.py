from langchain.llms import OpenAI
from langchain_mistralai import ChatMistralAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder  # Corrected import
import os
import json

# Initialize the MistralAI language model
api_key = os.environ.get('MISTRALAI_API_KEY')
if not api_key:
    raise ValueError(
        "MISTRALAI_API_KEY is not set in the environment variables.")
llm = ChatMistralAI(api_key=api_key)

# Load JSON files
emotion_loader = JSONLoader(file_path='data/emotion_log.json',
                            jq_schema='.[]',
                            text_content=False)
speech_loader = JSONLoader(file_path='data/speech_log.json',
                           jq_schema='.[]',
                           text_content=False)
slides_loader = JSONLoader(file_path='data/slides_log.json',
                           jq_schema='.[]',
                           text_content=False)

# Load documents
emotion_docs = emotion_loader.load()
speech_docs = speech_loader.load()
slides_docs = slides_loader.load()

# Combine all documents
all_docs = emotion_docs + speech_docs + slides_docs

# Split documents
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(all_docs)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# Create QA chain
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Define the system prompt
system_prompt = """Analyze the provided presentation analytics logs and answer the user's questions by extracting insights and leveraging your knowledge base.

# Steps

1. **Understand User's Question**: Carefully comprehend the specifics of the user's inquiry to understand what information they are seeking.
2. **Analyze the Logs**: Review the presentation analytics logs to identify relevant data points that pertain to the user's question.
3. **Leverage Knowledge**: Use your existing knowledge to contextualize the data, providing explanations that are informed by broader trends or best practices in presentation analytics.
4. **Formulate Insights**: Synthesize the log data and your background knowledge to generate comprehensive insights.
5. **Answer the Question**: Present a clear and concise answer to the user's question, ensuring it directly addresses their query with evidence from the logs and your contextual understanding.

# Output Format

Provide a well-structured paragraph addressing the user's question. Begin with a brief summary of the relevant analytics, followed by an explanation that ties in broader knowledge as necessary. End with a direct answer or insight based on the analysis.

# Examples

**Example 1:**
- **User's Question**: "What trends do you see in the engagement levels over the last three presentations?"
- **Insight**: "Analyzing the logs, the engagement levels show a consistent upward trend, with a 10% increase observed in each successive presentation. This could be attributed to increased interactivity and audience participation elements present in the latest presentations. Overall, this suggests an improvement in audience retention strategies."

**Example 2:**
- **User's Question**: "How does the viewership of the most recent presentation compare to historical averages?"
- **Insight**: "The most recent presentation saw a 15% higher viewership compared to the historical average. This can be linked to the recent social media campaign, proposing a direct correlation between increased marketing efforts and viewership boost."

# Notes

- Ensure that insights are specific, actionable, and evidence-backed.
- Consider factors like time of day, subject matter, audience size, and engagement tools when analyzing trends.
- You will not reference timestamps, but relative times. Instead of saying at 
- If relevant data is not present in the logs, use logical inferences where possible, but clearly state any assumptions made."""

# Define the prompt with 'history' and 'input'
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),  # Expects a list of messages
    ("user", "{input}"),  # Latest user input
])

# Initialize the conversation chain with memory set to return messages
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(
        return_messages=True),  # Enable message list
    verbose=True,
    prompt=prompt,
)

# Removed the redundant system message addition
# conversation.add_message(("system", system_prompt))


def get_chatbot_response(user_input):
    """
    Get a response from the chatbot based on the user input, conversation history, and log data.
    """
    # First, try to find relevant documents
    docs = vectorstore.similarity_search(user_input, k=4)

    if docs:
        # If relevant documents are found, use the QA chain to extract relevant information
        qa_response = qa_chain.run(input_documents=docs, question=user_input)

        # Incorporate the QA response into the conversation input
        combined_input = f"Relevant information: {qa_response}\n{user_input}"

        # Invoke the conversation chain with the combined input
        conversation_response = conversation.invoke(input=combined_input)

        # Extract the actual response text from the dictionary
        response_text = conversation_response['response']

        return response_text.strip()
    else:
        # If no relevant documents are found, use the conversation chain directly
        conversation_response = conversation.invoke(input=user_input)
        response_text = conversation_response['response']
        return response_text.strip()


# Example usage
if __name__ == "__main__":
    print(
        "Chatbot: Hello! I'm your presentation analytics assistant. How can I help you today?"
    )
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = get_chatbot_response(user_input)
        print(f"Chatbot: {response}")
