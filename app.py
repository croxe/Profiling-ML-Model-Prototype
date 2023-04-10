"""
Description
This is a ChatGPT to Data visualization App, by simple commands, everyone can get nice visualization graphics.
"""
# Core Pkgs
import streamlit as st
import os
import re
import argparse
import configparser
import base64
from io import BytesIO
# NLP Pkgs
import tiktoken
import openai, tenacity
from textblob import TextBlob
import spacy
from io import StringIO
from contextlib import redirect_stdout
import streamlit.components.v1 as components
import pandas as pd

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer



def chat_toCode(text, encoding, api_key, max_token_num = 4096, language = 'English'):
    openai.api_key = api_key
    summary_prompt_token = 1000
    text_token = len(encoding.encode(text))
    clip_text_index = int(len(text)*(max_token_num - summary_prompt_token)/text_token)
    clip_text = text[:clip_text_index]
    #need to change some of the message
    # add variable type to colmun
    messages=[
            {"role": "system", "content": "You are a professional programmer"},
            {"role": "assistant", "content": "This is the path of dataset: './Input/cpu-data.csv', with column name: [timestamp, system_value, last_idle, nice, irq, idle, last_total, utilization, iowait, username, total, softirq, upload_date, guest, guestnice, steal, tag, site, host, workload, iteration, time_stamp, is_workload_data, cpu, systems]. I need your help to write code for the following topic: "+clip_text},
            {"role": "user", "content": """  Write code with out explaination, and use mpld3 library to save the plot as "output.html"
             """},
        ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    result = ''
    for choice in response.choices:
        m = re.findall(r"```[^`]+```", choice.message.content)
        for i in m:
            result += i.strip("```python").strip("```")      
    return result  


def main():
    """ ChatGPT to VIS Based App with Streamlit """

    # Title
    st.title("ChatGPT to VIS Application")
    st.subheader("Natural Language Processing for everyone")
    st.markdown("""
    	#### Description
    	This is a ChatGPT to Data visualization App, by simple commands, everyone can get graphics with nice visualization.
    	""")
    dataset_name = './Input/cpu-data.csv'
    dataframe = pd.read_csv(dataset_name)
    st.markdown("""### Current dataframe columns are: 
    """ + str(list(dataframe.columns.values)))
    # Sentiment Analysis
    if st.checkbox("Get the visualization by text"):
        st.subheader("Put wanted plot in your Text")
        #import time
        #start_time = time.time()
        encoding = tiktoken.get_encoding("gpt2")
        api_key = "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"
        message = st.text_area("Enter Text","Type Here...")
        if st.button("Analyze"):
            code = chat_toCode(message, encoding, api_key)
            st.success(code)
            os.chdir('./')
            exec(code)
            with open('output.html', 'r') as file:
                html = file.read()
            components.html(html, height=600)
            
            
    
    # ReRun sample code
    if st.checkbox("Try to re-run code"):
        st.subheader("find out")

        message = st.text_area("Enter Text","Type Here.")
        if st.button("Rerun"):
            exec(message)
            with open('output.html', 'r') as file:
                html = file.read()
            components.html(html, height=600)
            #st.json(nlp_result)



    st.sidebar.subheader("About the App")
    st.sidebar.text("ChatGPT for data visualization.")
    st.sidebar.info("Use this tool to get data visualization as you wished.")
    st.sidebar.subheader("Developed by")
    st.sidebar.text("gs1")




if __name__ == '__main__':
    main()