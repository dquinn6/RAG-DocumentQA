import os
import logging
from logging.handlers import RotatingFileHandler
from config import config
import streamlit as st
from src.utils import update_patterns_json  

LOG_PATH = config.user_config["LOG_PATH"]

def btn_lock_callback():
    st.session_state.disable_flg = not st.session_state.disable_flg

# Direct UI logging to streamlit.log through logger object
def create_logger(
    name: str = "logger_",
    level: str = "INFO",
    filename: str = LOG_PATH+"streamlit.log",
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
    # If no handler present, add one
    if(
        sum([isinstance(handler, RotatingFileHandler) for handler in logger.handlers]) == 0
    ):
        handler = RotatingFileHandler(
            filename, maxBytes=max_log_size, backupCount=backup_count,
        )
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
    
    return logger

def render_doc_viewer(
        docs_name="documents", 
        index_name="current_index",
        include_text=None,
        unique_key="key",
    ):
    
    docviewer = st.empty() 
    idx_placeholder = st.empty()

    # Initialize the current index
    if index_name not in st.session_state:
        st.session_state[index_name] = 0

    # Reset if user filters some docs and current idx is out of bounds
    if st.session_state[index_name] > len(st.session_state[docs_name]) - 1:
        st.session_state[index_name] = 0

    show_next = st.button(
        label="Next document" if not st.session_state.disable_flg else "Disabled during process", 
        disabled=st.session_state.disable_flg,
        key=unique_key,
    )

    # Update index on button click
    if show_next:
        # Loop back to beginning if at end of list
        if st.session_state[index_name] >= len(st.session_state[docs_name]) - 1:
            st.session_state[index_name] = 0
        else:
            st.session_state[index_name] += 1

    with idx_placeholder.container(border=False):
        if len(st.session_state[docs_name]) != 0:
            st.write(f"Document: {st.session_state[index_name] + 1} / {len(st.session_state[docs_name])}")

    # Show next element in list
    with docviewer.container(height=300, border=True):
        if include_text:
            st.write(include_text)
        if len(st.session_state[docs_name]) == 0:
            st.write("No documents with this criteria could be found; try a larger token limit or different search pattern.")
        else:
            st.write(st.session_state[docs_name][st.session_state[index_name]])

def gather_docs(data_processor, communicator, search_pattern=None, replace_pattern=None, verbose=True):
    if search_pattern == None or replace_pattern == None:
        if verbose:
            st.sidebar.write("Please input search and replace patterns")
    else:
        update_patterns_json(key=search_pattern, val=replace_pattern)
        docs_with_pattern = data_processor.ret_passages_with_pattern(search_pattern)
        if verbose:
            st.sidebar.write(f"{len(docs_with_pattern)} / {len(data_processor.data)} documents with this search pattern.")
        docs_valid = [p for p in docs_with_pattern if communicator.count_tokens(p) < config.user_config["TOKEN_LIMIT"]]
        if verbose:
            st.sidebar.write(f"{len(docs_valid)} / {len(docs_with_pattern)} documents under token limit.")
        st.session_state["docs_valid"] = docs_valid
        st.session_state["docs_with_pattern"] = docs_with_pattern

def get_model_factory_name(model_name, rag=False):

    if not rag:
        names_mapped = {
            "gpt-3.5-turbo": "GPT_3.5_TURBO",
            "gpt-4": None,
            "gpt-4-32k": None
        }
        return names_mapped[model_name]
    
    else:
        names_mapped = {
            "gpt-3.5-turbo": "GPT_3.5_TURBO_RAG",
            "gpt-4": None,
            "gpt-4-32k": None
        }
        return names_mapped[model_name]