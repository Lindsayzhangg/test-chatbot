import streamlit as st
from langchain_voyageai import VoyageAIEmbeddings
import os
import boto3
from urllib.parse import urlparse
import pinecone
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, RetrievalQA
import time
import re
import json
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import uuid
import warnings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_pinecone import PineconeVectorStore
from langchain_voyageai import VoyageAIEmbeddings
from langchain.chains import create_history_aware_retriever
from datasets import Dataset
import numpy as np
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from ragas.metrics import context_relevancy, answer_relevancy, faithfulness, context_recall, answer_correctness, harmfulness
from ragas import evaluate

# Ignore all warnings
warnings.filterwarnings("ignore")

# Set up Streamlit app
st.set_page_config(page_title="Patient Chatbot", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
        """
    <style>
    .main {
        background-color: #f0f8ff;
        color: #2f4f4f;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton>button:hover,
    .stButton>button:active,
    .stButton>button:focus {
        background-color: #1e90ff; /* 将鼠标悬停和点击时的背景色改为蓝色 */
        color: white !important; /* 将鼠标悬停和点击时的文字颜色改为白色 */
    }
    .stTextInput>div>div>input {
        background-color: #f8f8f8;
        color: #2f4f4f;
        font-size: 16px;
        padding: 10px;
        border-radius: 8px;
        border: 2px solid #4CAF50;
    }
    .stTextInput>div>div>input:focus {
        border: 2px solid #1e90ff !important; 
        box-shadow: none !important; 
    }

    .stDownloadButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stDownloadButton>button:hover,
    .stDownloadButton>button:active,
    .stDownloadButton>button:focus {
        background-color: #1e90ff; 
        color: white !important; /* 确保点击后文字颜色为白色 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Patient Chatbot with Retrieval Abilities")

# Function to generate pre-signed URL
def generate_presigned_url(s3_client, s3_uri):
    parsed_url = urlparse(s3_uri)
    bucket_name = parsed_url.netloc
    object_key = parsed_url.path.lstrip('/')
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': object_key},
        ExpiresIn=3600
    )
    return presigned_url

# Function to retrieve documents, generate URLs, and format the response
def retrieve_and_format_response(user_input, retriever, llm):
    response = conversational_rag_chain.invoke({"input": user_input}, config={"configurable": {"session_id": "test"}})
    return response

# New function to save retrieved documents to a file without sensitive information
def save_retrieved_docs_to_file(docs):
    retrieved_docs_content = []
    for doc in docs:
        retrieved_docs_content.append({
            "content": doc.page_content,
            "more_info": f"URL_PLACEHOLDER_{doc.metadata['id']}"
        })
    return json.dumps(retrieved_docs_content, indent=2)

# Function to dynamically replace URL placeholders with URLs
def replace_placeholders_with_urls(content, s3_client):
    data = json.loads(content)
    for item in data:
        s3_uri = item["more_info"].replace("URL_PLACEHOLDER_", "")
        presigned_url = generate_presigned_url(s3_client, s3_uri)
        item["more_info"] = presigned_url
    return json.dumps(data, indent=2)

# Setup - Streamlit secrets
OPENAI_API_KEY = st.secrets["api_keys"]["OPENAI_API_KEY"]
VOYAGE_AI_API_KEY = st.secrets["api_keys"]["VOYAGE_AI_API_KEY"]
PINECONE_API_KEY = st.secrets["api_keys"]["PINECONE_API_KEY"]
aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
aws_region = st.secrets["aws"]["aws_region"]

# Initialize boto3 S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# Pull the retrieval QA chat prompt
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
groundtruth_llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)

# Initialize the retriever using PineconeVectorStore
model_name = "voyage-large-2"
embedding_function = VoyageAIEmbeddings(
    model=model_name,
    voyage_api_key= VOYAGE_AI_API_KEY
)
vector_store = PineconeVectorStore.from_existing_index(
    embedding=embedding_function,
    index_name="test"
)
retriever = vector_store.as_retriever()

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

# Groundtruth retriever chain
gt_history_aware_retriever = create_history_aware_retriever(
    groundtruth_llm, retriever, contextualize_q_prompt
)
gt_question_answer_chain = create_stuff_documents_chain(groundtruth_llm, retrieval_qa_chat_prompt)
gt_rag_retreival_chain = create_retrieval_chain(gt_history_aware_retriever, gt_question_answer_chain)

# Initialize memory
memory = ConversationBufferMemory()

# Store for chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
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

gt_conversational_rag_chain = RunnableWithMessageHistory(
    gt_rag_retreival_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def retrieve_and_format_response(user_input, retriever, llm):
    response = conversational_rag_chain.invoke({"input": user_input}, config={"configurable": {"session_id": "test"}})
    return response

def gt_retrieve_and_format_response(user_input, retriever, llm):
    response = gt_conversational_rag_chain.invoke({"input": user_input}, config={"configurable": {"session_id": "test"}})
    return response

# BLEU score calculation
def bleu_score(reference, hypothesis):
    reference_tokens = [nltk.word_tokenize(reference)]
    hypothesis_tokens = nltk.word_tokenize(hypothesis)
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothing_function)

# Edit distance calculation
def edit_distance(reference, hypothesis):
    m = len(reference) + 1
    n = len(hypothesis) + 1

    # Create a matrix to store the distances
    dp = np.zeros((m, n), dtype=int)

    # Initialize the first row and column
    for i in range(m):
        dp[i][0] = i
    for j in range(n):
        dp[0][j] = j

    # Compute the edit distance
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if reference[i-1] == hypothesis[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,        # Deletion
                           dp[i][j-1] + 1,        # Insertion
                           dp[i-1][j-1] + cost)   # Substitution

    return dp[m-1][n-1]

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
        bot_response = retrieve_and_format_response(user_input, retriever, llm)["answer"]
        gt_bot_response = gt_retrieve_and_format_response(user_input, retriever, groundtruth_llm)["answer"]
    
    st.session_state["messages"].append({"role": "assistant", "content": bot_response})
    
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    
    # Retrieve documents and prepare the content for download
    docs = retriever.get_relevant_documents(user_input)  # Use retriever to get documents
    retrieved_docs_content = save_retrieved_docs_to_file(docs)
    
    # Dynamically replace URL placeholders with pre-signed URLs
    retrieved_docs_content = replace_placeholders_with_urls(retrieved_docs_content, s3_client)
    
    # Display a download button for the retrieved documents
    st.download_button(
        label="More Info",
        data=retrieved_docs_content,
        file_name="retrieved_documents.json",
        mime="application/json"
    )
    
    # Prepare data for evaluation
    eval_data = {
        "question": user_input,
        "contexts": [doc.page_content for doc in docs],
        "answer": bot_response,
        "ground_truth": gt_bot_response
    }
    
    # Create a Dataset for evaluation
    dataset_eval = Dataset.from_pandas(pd.DataFrame([eval_data]))

    # Evaluate the conversation data
    result = evaluate(
        dataset_eval,
        metrics=[
            context_relevancy,
            faithfulness,
            answer_relevancy,
            context_recall,
            harmfulness,
            answer_correctness
        ],
    )
    eval_df = result.to_pandas()

    # Print evaluation metrics
    st.markdown(f"**BLEU score**: {round(bleu_score(gt_bot_response, bot_response), 6)}")
    st.markdown(f"**Edit distance**: {edit_distance(gt_bot_response, bot_response)}")
    st.markdown(f"**Context relevancy**: {round(eval_df.context_relevancy.loc[0], 6)}")
    st.markdown(f"**Faithfulness**: {eval_df.faithfulness.loc[0]}")
    st.markdown(f"**Answer relevancy**: {round(eval_df.answer_relevancy.loc[0], 6)}")
    st.markdown(f"**Answer correctness**: {eval_df.answer_correctness.loc[0]}")
    st.markdown(f"**Context recall**: {round(eval_df.context_recall.loc[0], 6)}")
    st.markdown(f"**Harmfulness**: {round(eval_df.harmfulness.loc[0], 6)}")

# Add an "End Conversation" button
if st.button("End Conversation"):
    # Save chat history to a file and upload to S3
    session_id = str(uuid.uuid4())
    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]])
    local_filename = f"chat_history_{session_id}.txt"
    with open(local_filename, 'w') as file:
        file.write(chat_history)
    chat_history_key = f"raw-data/chat_history_{session_id}.txt"
    s3_client.upload_file(local_filename, "chat-history-process", chat_history_key)
    st.success(f"Chat history saved and uploaded to S3 as '{chat_history_key}'")
    # Clear chat history from session state
    st.session_state["messages"] = []
