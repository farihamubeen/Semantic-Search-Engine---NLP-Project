import numpy as np
import faiss
import requests

from sentence_transformers import SentenceTransformer, CrossEncoder, util
import os, re
import torch
import string
import numpy as np
import pandas as pd
import validators
import nltk
import warnings
import streamlit as st
from PIL import Image
import html


nltk.download('punkt')
from  nltk import sent_tokenize
warnings.filterwarnings("ignore")


def search_faiss(query):
    df = pd.read_csv(corpus + '.csv')
    index2=faiss.read_index(corpus+'_index')
    k = 20
    xq = model.encode([query])
    D, I = index2.search(xq, k)
    df_result = pd.DataFrame(columns=['IP_distance', 'filename'])
    df_result1 = pd.DataFrame(columns=['File_Name', 'File_Content'])
    I_result = I[0].tolist()
    D_result = D[0].tolist()

    for i in range(len(D_result)):
        df_result.loc[i, 'IP_distance'] = D_result[i]
        df_result.loc[i, 'filename'] = df.iloc[I_result[i]]['filename']
        #df_result.loc[i, 'Page_No'] = df.iloc[I_result[i]]['page_no']
        #url_srt = df.iloc[I_result[i]]['hyperlink']
        #df_result.loc[i, 'Hyperlink'] = f'<a target ="_blank" href = "{url_srt}">Document Page Link</a>'

        df_result1.loc[i, 'File_Name'] = df.iloc[I_result[i]]['filename']
        df_result1.loc[i, 'File_Content'] = df.iloc[I_result[i]]['file_content']

    st.write(df_result.to_html(escape=False, index=False), unsafe_allow_html=True)

    my_expander = st.expander(label='Expand me')
    with my_expander:
        st.write(df_result1.to_html(index=False), unsafe_allow_html=True)
        clicked = st.button('Click me!')


st.title(" Enterprise Semantic Search Engine ")

#st.subheader(f"A Multi Corpus-Multi Documents Semantic Documents Search Engine ")

st.markdown("Please choose the corpus on which search has to be performed. ")

st.markdown("eg: Safety Information Bulletins or Safety Directives or Airworthiness Directives or EASA ...etc.. ")

corpus_selection = ["test_corpus", "corpus"]

corpus = st.selectbox("Choose the corpus to be searched ...", options=corpus_selection, key='sbox2')


def clear_text():
    st.session_state["text_url"] = " "
    st.session_state["text_input"] = " "


def clear_search_text():
    st.session_state["text_input"] = " "
    st.session_state["text_url"] = " "


search_query = st.text_input("Please Enter your search query here", value="Aviation safety rules", key="text_input")
st.markdown("Higher the positive score more the content is relative to the query and vice versa.")
st.markdown("Score : +ve higher score indicates the query and the document content are more relative")
st.markdown("Score : 0 Content relevant to the query not found in the corpus")
st.markdown("Score : -ve Lower score indicates the documents are not related to teh query")

col1, col2 = st.columns(2)

with col1:
    search = st.button("Search", key='search_but', help='Click to Search!!')

with col2:
    Clear = st.button("Clear text", on_click=clear_text, key='clear',
                      help='Click to clear the URL input and search query')

if search:

    with st.spinner(
            text=f"Loading embeddings. This might take a few seconds depending on the length of your document..."
    ):
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-dot-prod-v3')
        if torch.cuda.is_available():
            model = model.to(torch.device("cuda"))
            print(model.device)

        #model = SentenceTransformer('sentence-transformers_msmarco-distilbert-base-dot-prod-v3')
    with st.spinner(
            text="Embeddings completed, searching for relevant text for given query..."):
        st.markdown("The top results")
        search_faiss(search_query)
