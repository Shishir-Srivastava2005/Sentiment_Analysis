import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.title("My Sentiment Analysis Data")

df = pd.read_csv("final_df.csv") 

st.subheader("Dataframe Preview")
st.dataframe(df.head())

st.subheader("EDA")
st.image("Score_count.png", caption="Original Score Distribution")
st.image("CorrMatrix.png", caption="Correlation Matrix")

col1, col2 = st.columns(2)
with col1:
    st.write("Wordcloud of Positive Reviews")
    st.image("pos_wc.png")
with col2:
    st.write("Wordcloud of Negative Reviews")
    st.image("neg_wc.png")

st.subheader("VADER Model v/s BERT Model")

v_report = pd.read_csv("vader_report.csv", index_col=0)
b_report = pd.read_csv("bert_report.csv", index_col=0)

col3, col4 = st.columns(2)
with col3:
    st.subheader("VADER Report")
    st.dataframe(v_report, use_container_width=True)
with col4:
    st.subheader("BERT Report")
    st.dataframe(b_report, use_container_width=True)

col5, col6 = st.columns(2)
with col5:
    st.write("Confusion Matrix for VADER")
    st.image("confusionv.png")
with col6:
    st.write("Confusion Matrix for BERT")
    st.image("confusion.png")