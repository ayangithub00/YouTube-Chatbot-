import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="YouTube Chatbot", page_icon="🎥")

st.title("🎥 YouTube Transcript Chatbot")
st.write("Ask questions about any YouTube video transcript.")


video_id = st.text_input(
    "Enter YouTube Video ID",
    value="m7v7KLttM3k"
)

question = st.text_input(
    "Ask a question about the video"
)


def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


if st.button("Ask"):

    try:
        with st.spinner("Fetching transcript..."):

            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.fetch(
                video_id,
                languages=["en"]
            )

            transcript = " ".join(
                chunk.text
                for chunk in transcript_list
            )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.create_documents([transcript])

            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5"
            )

            vector_store = FAISS.from_documents(
                chunks,
                embeddings
            )

            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )

            llm = HuggingFaceEndpoint(
                repo_id="Qwen/Qwen2.5-72B-Instruct",
                task="text-generation",
                max_new_tokens=512,
                temperature=0.5,
            )

            model = ChatHuggingFace(llm=llm)

            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.

                Answer ONLY from the provided transcript context.

                If the context is insufficient,
                say "I don't know."

                Context:
                {context}

                Question:
                {question}
                """,
                input_variables=["context", "question"]
            )

            chain = (
                RunnableParallel(
                    {
                        "context": retriever
                        | RunnableLambda(format_docs),
                        "question": RunnablePassthrough(),
                    }
                )
                | prompt
                | model
                | StrOutputParser()
            )

            answer = chain.invoke(question)

            st.subheader("Answer")
            st.write(answer)

    except TranscriptsDisabled:
        st.error("No transcript available for this video.")

    except Exception as e:
        st.error(str(e))