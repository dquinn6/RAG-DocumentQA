import os
import logging
from logging.handlers import RotatingFileHandler
from config import config
import streamlit as st
import json
from src.communicators import GPTCommunicator
from src.data_processors import WikiTextProcessor
from src.vectorstore_handlers import LangchainVectorstore
from langchain_community.embeddings import HuggingFaceEmbeddings

with open('config/manipulate_patterns.json') as f:
    PATTERNS = list(json.load(f).items())
 
API_KEY = config.user_config["ACCESS_TOKEN"]
MODEL_NAME = config.user_config["MODEL_NAME"]
SAVE_PATH = config.user_config["SAVE_PATH"]
SEARCH_TYPE = config.user_config["SEARCH_TYPE"]
N_RETRIEVED_DOCS = config.user_config["N_RETRIEVED_DOCS"]
TOKEN_LIMIT = config.user_config["TOKEN_LIMIT"]
VERBOSE = config.user_config["VERBOSE"]

LOG_PATH = "logs/"

def create_logger(
    name: str = "logger_",
    level: str = "DEBUG",
    filename: str = "streamlit.log",
    max_log_size: int = 1024 * 1024 * 1024, #100 MB
    backup_count: int = 1,
):
    """ Create logger for handling streamlit IO

    Solution to duplicate log messages; found at https://discuss.streamlit.io/t/streamlit-duplicates-log-messages-when-stream-handler-is-added/16426

    Args:
        name (str, optional): _description_. Defaults to "logger_".
        level (str, optional): _description_. Defaults to "DEBUG".
        filename (str, optional): _description_. Defaults to "app.log".
        max_log_size (int, optional): _description_. Defaults to 1024*1024*1024.

    Returns:
        _type_: _description_
    """
    if not os.path.exists(LOG_PATH):
        os.mkdir(LOG_PATH)
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    # if no handler present, add one
    if(
        sum([isinstance(handler, RotatingFileHandler) for handler in logger.handlers]) == 0
    ):
        handler = RotatingFileHandler(
            LOG_PATH + filename, maxBytes=max_log_size, backupCount=backup_count,
        )
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
    
    return logger


@st.cache_resource #run only once
def init_RAG():

    communicator = GPTCommunicator(api_key=API_KEY, model_name="gpt-3.5-turbo")

    processor = WikiTextProcessor(
        dataset_version = "wikitext-2-raw-v1", 
        split = "train", 
        communicator = communicator,
        verbose = VERBOSE
    )
    _ = processor.process_text(
        token_limit = TOKEN_LIMIT, 
        save_path = SAVE_PATH,
        save_filename = "processed_data.csv",
        manipulate_pattern = PATTERNS
    )

    vs = LangchainVectorstore(
        embedding_type = HuggingFaceEmbeddings(),
        processed_csv_path = SAVE_PATH+"processed_data.csv",
        verbose = VERBOSE
    )
    if not os.path.exists(SAVE_PATH):
        vs.create_local_vectorstore(save_path=SAVE_PATH)
    else:
        vs.load_local_vectorstore(load_path=SAVE_PATH)

    vs.create_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": N_RETRIEVED_DOCS
        }
    )

    communicator.set_vectorstore_handler(vs)

    return communicator


def run_streamlit_app():

    st.set_page_config(page_title="Page")

    communicator = init_RAG()

    if "logger" not in st.session_state:
        st.session_state["logger"] = create_logger()
    logger = st.session_state["logger"]

    st.title("Title")

    st.header("header")

    st.subheader("subheader")

    selected = st.selectbox(
        "Select an option:",
        ["Select"] + ["Option 1", "Option 2"]
    )

    logger.info(f"selected: {selected}")

    user_query = st.text_input("Query: ")

    if st.button("Get Answer"):
        if user_query:

            # do rag
            response, retrieved_context = communicator.post_rag_prompt(user_query)

            st.subheader("Response:")
            st.write(response)

            logger.info(f"User query: {user_query}")

            # Get user feedback

            feedback = st.radio(
                "Select an option:",
                [
                    "None",
                    "Yes",
                    "No",
                ],
            )
            if feedback != "None":
                logging.info(feedback)


run_streamlit_app()
