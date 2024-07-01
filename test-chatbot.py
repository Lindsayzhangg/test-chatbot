import streamlit as st
import os
import boto3
from urllib.parse import urlparse
import pinecone
from langchain_openai import ChatOpenAI
from langchain.chains import create_stuff_documents_chain, create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_voyageai import VoyageAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import uuid
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Set up Streamlit app
st.set_page_config(page_title="Custom Chatbot", layout="wide")
st.title("Custom Chatbot with Retrieval Abilities")

# Function to generate pre-signed URL
def generate_presigned_url(s3_uri):
    parsed_url = urlparse(s3_uri)
    bucket_name = parsed_url.netloc
    object_key = parsed_url.path.lstrip('/')
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': object_key},
        ExpiresIn=3600
    )
    return presigned_url

# Setup Streamlit secrets
OPENAI_API_KEY = st.secrets["api_keys"]["OPENAI_API_KEY"]
VOYAGE_AI_API_KEY = st.secrets["api_keys"]["VOYAGE_AI_API_KEY"]
PINECONE_API_KEY = st.secrets["api_keys"]["PINECONE_API_KEY"]
aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
aws_region = st.secrets["aws"]["aws_region"]

# Initialize the retriever using PineconeVectorStore
model_name = "voyage-large-2"
embedding_function = VoyageAIEmbeddings(
    model=model_name,
    voyage_api_key=VOYAGE_AI_API_KEY
)
vector_store = PineconeVectorStore.from_existing_index(
    embedding=embedding_function,
    index_name="test"
)
retriever = vector_store.as_retriever()

# Pull the retrieval QA chat prompt
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)

# Create the combined documents chain
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
rag_retreival_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Initialize memory
memory = ConversationBufferMemory()

# CODE DIRECTLY FROM LANGCHAIN DOCUMENTATION
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_retreival_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Function to save chat history and upload to S3
def save_chat_history_to_file(filename, history):
    with open(filename, 'w') as file:
        file.write(history)

def upload_file_to_s3(s3_client, bucket, key, filename):
    s3_client.upload_file(filename, bucket, key)

def chat():
    session_id = str(uuid.uuid4())
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    bucket_name = "chat-history-process"
    chat_history_key = f"raw-data/chat_history_{session_id}.txt"
    chat_history = f"\nSession ID: {session_id}\n"
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ending the conversation. Goodbye!")
            break
        response = conversational_rag_chain.invoke({"input": user_input}, config={"configurable": {"session_id": "test"}})["answer"]
        print("Bot:", response)
        
        # Append interaction to chat history
        chat_history += f"You: {user_input}\nAI: {response}\n"
    
    # Save the chat history to a file
    local_filename = f"./history/chat_history_{session_id}.txt"
    save_chat_history_to_file(local_filename, chat_history)

    # Upload the file to S3
    upload_file_to_s3(s3_client, bucket_name, chat_history_key, local_filename)
    print(f"Chat history saved and uploaded to S3 as '{chat_history_key}' in bucket '{bucket_name}'")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages from history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_input = st.chat_input("You: ")

if user_input:
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and display bot response
    with st.spinner("Thinking..."):
        response = conversational_rag_chain.invoke({"input": user_input}, config={"configurable": {"session_id": "test"}})["answer"]
    
    st.session_state["messages"].append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.markdown(response)

# Add an "End Conversation" button
if st.button("End Conversation"):
    # Save chat history to a file and upload to S3
    session_id = str(uuid.uuid4())
    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]])
    local_filename = f"chat_history_{session_id}.txt"
    save_chat_history_to_file(local_filename, chat_history)
    chat_history_key = f"raw-data/chat_history_{session_id}.txt"
    upload_file_to_s3(s3_client, "chat-history-process", chat_history_key, local_filename)
    st.success(f"Chat history saved and uploaded to S3 as '{chat_history_key}'")
    # Clear chat history from session state
    st.session_state["messages"] = []
