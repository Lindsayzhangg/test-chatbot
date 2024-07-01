import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_voyageai import VoyageAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from urllib.parse import urlparse
import boto3
import uuid
import re
import pinecone
import openai
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

# 从 secrets.toml 文件中加载环境变量
try:
    OPENAI_API_KEY = st.secrets["api_keys"]["OPENAI_API_KEY"]
    VOYAGE_AI_API_KEY = st.secrets["api_keys"]["VOYAGE_AI_API_KEY"]
    PINECONE_API_KEY = st.secrets["api_keys"]["PINECONE_API_KEY"]
    AWS_ACCESS_KEY_ID = st.secrets["aws"]["aws_access_key_id"]
    AWS_SECRET_ACCESS_KEY = st.secrets["aws"]["aws_secret_access_key"]
    AWS_REGION = st.secrets["aws"]["aws_region"]
except KeyError as e:
    st.error(f"Missing key in secrets.toml: {e}")
    raise

# 初始化OpenAI
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4")

# 初始化VoyageAI嵌入
model_name = "voyage-large-2"
embedding_function = VoyageAIEmbeddings(
    model=model_name, 
    voyage_api_key=VOYAGE_AI_API_KEY
)

# 初始化Pinecone向量存储
pinecone.init(api_key=PINECONE_API_KEY)
vector_store = PineconeVectorStore.from_existing_index(
    embedding=embedding_function,
    index_name="test"
)
retriever = vector_store.as_retriever()

# 定义系统提示和问答提示
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 正确的创建问答链
question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt,
)

# 使用 create_retrieval_chain 创建历史记录感知检索器
history_aware_retriever = create_retrieval_chain(retriever, question_answer_chain)

# 初始化对话历史
memory = ConversationBufferMemory()
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    history_aware_retriever,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# 生成预签名URL
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

# 检索和格式化响应
def retrieve_and_format_response(query, retriever, llm):
    docs = retriever.get_relevant_documents(query)
    formatted_docs = []
    for doc in docs:
        content_data = doc.page_content
        s3_uri = doc.metadata['id']
        s3_gen_url = generate_presigned_url(s3_client, s3_uri)
        formatted_doc = f"{content_data}\n\n[More Info]({s3_gen_url})"
        formatted_docs.append(formatted_doc)
    combined_content = "\n\n".join(formatted_docs)
    prompt = f"""Instruction: You are a helpful assistant to help users with their queries. Based on the following information, provide a summarized & concise explanation using a couple of sentences. Only respond with the information relevant to the user query {query}. If there are none, say 'I don't know'. Context: {combined_content}"""
    message = HumanMessage(content=prompt)
    response = llm([message])
    return response

# 保存对话历史到文件
def save_chat_history_to_file(filename, history):
    with open(filename, 'w') as file:
        file.write(history)

# 上传文件到S3
def upload_file_to_s3(s3_client, bucket, key, filename):
    s3_client.upload_file(filename, bucket, key)

# Streamlit设置
st.set_page_config(page_title="Custom Chatbot", layout="wide")
st.title("Custom Chatbot with Retrieval Abilities")

# 初始化S3客户端
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# 初始化会话历史
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 显示历史消息
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 获取用户输入
user_input = st.chat_input("You: ")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.spinner("Thinking..."):
        bot_response = retrieve_and_format_response(user_input, retriever, llm).content
    st.session_state["messages"].append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)

# 添加“结束对话”按钮
if st.button("End Conversation"):
    session_id = str(uuid.uuid4())
    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]])
    local_filename = f"chat_history_{session_id}.txt"
    save_chat_history_to_file(local_filename, chat_history)
    chat_history_key = f"raw-data/chat_history_{session_id}.txt"
    upload_file_to_s3(s3_client, "chat-history-process", chat_history_key, local_filename)
    st.success(f"Chat history saved and uploaded to S3 as '{chat_history_key}'")
    st.session_state["messages"] = []
