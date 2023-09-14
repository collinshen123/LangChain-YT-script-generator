import os
from dotenv import load_dotenv
import streamlit as st
import time
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

load_dotenv()
openai_api_key = os.environ['OPENAI_API_KEY'] 

# # os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']


st.title('🫵🚇Youtube Video Script Generator')
st.write("")
prompt = st.text_input('Enter a prompt for the AI to complete')



#prompt template
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'write me a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template = 'write me a youtube video script about this title: {title} while leveraging this wikipedia research: {wikipedia_research}'
)

title_memory = ConversationBufferMemory(input_key='topic', memory_key='title_memory')
script_memory = ConversationBufferMemory(input_key='title', memory_key='script_memory')


llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm = llm, prompt = title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm = llm, prompt = script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

st.sidebar.title('Script Generation Details')

if prompt:
    
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    st.write("")
    st.write(
        '<span style="font-size: 20px; text-decoration: underline;">Title:</span>',
        unsafe_allow_html=True,
    )
    st.write(title)
    st.write("")
    st.write(
        '<span style="font-size: 20px; text-decoration: underline;">Script:</span>',
        unsafe_allow_html=True,
    )
    st.write(script)
    st.write("") 

    with st.sidebar.expander('Wikipedia Research'):
        st.info(wiki_research)

    with st.sidebar.expander('Title history'):
        st.info(title_memory.buffer)
    
    with st.sidebar.expander('Script history'):
        st.info(script_memory.buffer)
    
    st.success("Script Generated Successfully!")
    st.write("")

    
        
        
