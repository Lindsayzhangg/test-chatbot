import streamlit as st
from langchain_voyageai import VoyageAIEmbeddings
import os
import boto3
from urllib.parse import urlparse
from pinecone import Pinecone
import pinecone
from langchain_openai import ChatOpenAI
import openai
from langchain.chains import LLMChain, RetrievalQA
import time
import re
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import uuid
import warnings
import os
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain_voyageai import VoyageAIEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
import uuid
import boto3

# Ignore all warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

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

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY)

# Initialize OpenAI and VoyageAI
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)
embedding_function = VoyageAIEmbeddings(
    model="voyage-large-2",
    voyage_api_key=VOYAGE_AI_API_KEY
)

# Initialize the retriever using PineconeVectorStore
vector_store = PineconeVectorStore.from_existing_index(
    embedding=embedding_function,
    index_name="test"
)
retriever = vector_store.as_retriever()

# Set up contextualize question prompt
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

# Set up QA prompt
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
question_answer_chain = create_retrieval_chain(history_aware_retriever, qa_prompt)

# Initialize memory
memory = ConversationBufferMemory()

# Initialize chat message history store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    question_answer_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Function to save chat history to a file
def save_chat_history_to_file(filename, history):
    with open(filename, 'w') as file:
        file.write(history)

# Function to upload file to S3
def upload_file_to_s3(s3_client, bucket, key, filename):
    s3_client.upload_file(filename, bucket, key)

# Chat function
def chat():
    print("Start chatting with the bot (type 'exit' to stop):")
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
        response = conversational_rag_chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})["answer"]
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
        bot_response = conversational_rag_chain.invoke({"input": user_input}, config={"configurable": {"session_id": str(uuid.uuid4())}})["answer"]

    st.session_state["messages"].append({"role": "assistant", "content": bot_response})

    with st.chat_message("assistant"):
        st.markdown(bot_response)

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
