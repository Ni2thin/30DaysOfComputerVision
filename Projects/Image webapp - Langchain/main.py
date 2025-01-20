from tempfile import NamedTemporaryFile
import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from tools import ImageCaptionTool, ObjectDetectionTool

tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=1,
    return_messages=True
)

llm = ChatOpenAI(
    openai_api_key='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',
    temperature=0,
    model_name="gpt-3.5-turbo"
)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=1,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method='generate'
)

@st.cache_resource
def get_agent():
    return agent

@st.cache_data
def process_image_and_respond(user_question, image_path):
    agent = get_agent()
    return agent.run('{}, this is the image path: {}'.format(user_question, image_path))

# set title
st.title('Ask a question to an image')

# set header
st.header("Please upload an image")

# upload file
file = st.file_uploader("", type=["jpeg", "jpg", "png"])

if file:
    # display image
    st.image(file, use_column_width=True)

    # text input
    user_question = st.text_input('Ask a question about your image:')

    if user_question and user_question != "":
        with NamedTemporaryFile(dir='.') as f:
            f.write(file.getbuffer())
            image_path = f.name

            # write agent response
            with st.spinner(text="In progress..."):
                response = process_image_and_respond(user_question, image_path)
                st.write(response)
