import os
import time
import streamlit as st
from typing_extensions import List, Tuple
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header("Cocktail Advisor ChatBot")


def create_llm_and_embeddings():
    """With Google API"""
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro"), GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def load_data(file_path) -> List[Document]:
    loader = CSVLoader(file_path=file_path)
    data = loader.load_and_split()
    return data


def get_pinecone_stores(embeddings) -> Tuple[PineconeVectorStore, PineconeVectorStore]:
        def create_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pc = Pinecone(api_key=pinecone_api_key)
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        if "cocktails" not in existing_indexes:
            create_index("cocktails")
            docs = load_data("cocktails.csv")
            cocktails_vector_store = PineconeVectorStore.from_documents(
                docs, index_name="cocktails", embedding=embeddings
            )
        else:
            cocktails_vector_store = PineconeVectorStore(index_name="cocktails", embedding=embeddings)

        if "user-memories" not in existing_indexes:
            create_index("user-memories")
        user_vector_store = PineconeVectorStore(index_name="user-memories", embedding=embeddings)

        return cocktails_vector_store, user_vector_store


def create_agent():
    @tool
    def cocktails_retrieve(query: str):
        """Retrieve information about cocktails from local document database."""
        retrieved_docs = cocktails_vector_store.similarity_search(query, k=20)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\n" f"Content: {doc.page_content}"
            for doc in retrieved_docs
        )
        return retrieved_docs, serialized

    @tool
    def user_mem_retrieve(query: str):
        """Retrieve information about user favorite cocktails and ingredients."""
        retrieved_docs = user_vector_store.similarity_search(query, k=10)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\n" f"Content: {doc.page_content}"
            for doc in retrieved_docs
        )
        return retrieved_docs, serialized

    system_message = "You are a cocktail advisor."
    langgraph_agent_executor = create_react_agent(
        llm, [user_mem_retrieve, cocktails_retrieve], state_modifier=system_message
    )

    return langgraph_agent_executor


from pydantic import BaseModel, Field
class Preferences(BaseModel):
    """Information about user favorite cocktails and ingredients."""
    name: str = Field(
        ..., description="Name of cocktail or ingredient"
    )
    type: str = Field(
        ..., description="Is it ingredient or cocktail?"
    )
    attitude: str = Field(
        ..., description="How does the user feel about it?"
    )

class ExtractionData(BaseModel):
    """Extracted information about  user favorite cocktails or ingredients."""
    favorite_things: List[Preferences]


def create_extractor():
    """Optional model to extract user preferences from user input"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at identifying user favorite ingredients and cocktails in text."
                "Only extract favorite ingredients and cocktails. Extract nothing if no important information can be found in the text.",
            ),
            ("human", "{text}"),
        ]
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    extractor = prompt | llm.with_structured_output(
        schema=ExtractionData,
        include_raw=False,
    )
    return extractor


def update_user_memories(user_input, detecting=False):
    if not detecting:
        document = Document(
            page_content=user_input,
            metadata={"source": "user"},
        )
        user_vector_store.add_documents(documents=[document])
    else:
        result = extractor.invoke(user_input)
        for preference in result.favorite_things:
            document = Document(
                page_content=str(preference.content),
                metadata={"source": "user"},
            )
            user_vector_store.add_documents(documents=[document])



llm, embeddings = create_llm_and_embeddings()
cocktails_vector_store, user_vector_store = get_pinecone_stores(embeddings)
agent = create_agent()
extractor = create_extractor()


if __name__ == "__main__":
    query = st.chat_input("Enter your questions here")
    if "user_query_history" not in st.session_state:
        st.session_state["user_query_history"] = []
    if "chat_answers_history" not in st.session_state:
        st.session_state["chat_answers_history"] = []

    if query:
        with st.spinner("Generating......"):
            update_user_memories(query)
            output = agent.invoke({"messages": [("user", query)]})["messages"][-1].content
            st.session_state["chat_answers_history"].append(output)
            st.session_state["user_query_history"].append(query)

    if st.session_state["chat_answers_history"]:
        for i, j in zip(st.session_state["chat_answers_history"], st.session_state["user_query_history"]):
            message1 = st.chat_message("user")
            message1.write(j)
            message2 = st.chat_message("assistant")
            message2.write(i)
