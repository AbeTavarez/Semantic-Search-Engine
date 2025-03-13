import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

st.title("Semantic Search Engine")
st.header("Upload a file to get started.", divider="green")

# text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

# Embedding Model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Vector Store
chroma_vector_store = Chroma(
    collection_name="my_docs",
    embedding_function=embedding_model,
    persist_directory="./chroma/db"
)

# LLM Model
llm = ChatOpenAI(model="gpt-4o-mini")


uploaded_file = st.file_uploader("Select a file:")


if uploaded_file is not None:
    with st.spinner("Processing file..."):
        try:
            print("File info: ", uploaded_file)
            
            # save file in memory
            temp_file_path = uploaded_file.name
            
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # PDF file loader
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            # print("Docs: ", docs)
            
            # create chunks
            chunks = text_splitter.split_documents(docs)
            print("Chunks created: ", len(chunks))
            
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i} is of size ", len(chunk.page_content))
                
            # create embeddings
            # emb1 = embedding_model.embed_query(chunks[0].page_content)
            # print(emb1)
            
            # Index embedding
            chroma_ids = chroma_vector_store.add_documents(documents=chunks)
            print("Chroma Ids: ", chroma_ids)
            
            
            # Similarity Search
            # result = chroma_vector_store.similarity_search(
            #     "what is the main topic of the case study?"
            # )
            # print(result)
            
            retriever = chroma_vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 1}
            )
            
            if prompt := st.chat_input("Prompt"):
                print(prompt)
                
                docs_retrieved = retriever.invoke(prompt)
                
                # Create a Prompt Template
                system_prompt = """You're a helpful assistant. Please answer the following question {question}, only using the following information {document}.
                If you can't answer the question, just say you can't answer that.
                """
                
                prompt_template = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt)
                    ]
                )
                
                final_prompt = prompt_template.invoke({
                    "question": prompt,
                    "document": docs_retrieved
                })
                
                print("Final Prompt", final_prompt)
                
                # UI container
                result_placeholder = st.empty()
                
                # Create completion
                # completion = llm.invoke(final_prompt)
                # print("Completion", completion.content)
                
                # Streaming the completion result
                full_completion = ""
                for chunk in llm.stream(final_prompt):
                    full_completion += chunk.content
                    result_placeholder.write(full_completion)
            
        
        except Exception as e:
            print(e)
            
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)