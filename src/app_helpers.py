""" Module for UI application helper functions.  """

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

import streamlit as st
from src.config import config

from src import communicators, data_processors
from src.utils import update_patterns_json

# Define LOG_PATH in program config.yml
LOG_PATH = config.user_config["LOG_PATH"]


# Direct UI logging to streamlit.log through logger object
def create_logger(
    name: str = "logger_",
    level: str = "INFO",
    filename: str = LOG_PATH + "streamlit.log",
    max_log_size: int = 1024 * 1024 * 1024,  # 100 MB
    backup_count: int = 1,
) -> logging.Logger:
    """Create a RotatingFileHandler logger for handling streamlit IO. Expects variable LOG_PATH already defined in module.

    Solution to duplicate log messages; found at https://discuss.streamlit.io/t/streamlit-duplicates-log-messages-when-stream-handler-is-added/16426

    Args:
        name (str, optional): Name for logging.getLogger(). Defaults to "logger_".
        level (str, optional): Logging level to display messages. Defaults to "INFO".
        filename (str, optional): LOG_PATH (module variable) + filename to write log messages to. Defaults to LOG_PATH+"streamlit.log"
        max_log_size (int, optional): Max size (bytes) for log file before rollover. Defaults to 1024*1024*1024.
        backup_count (int, optional): number of old log files to maintain. Defaults to 1.

    Returns:
        logging.Logger: Logger object for UI message logging.
    """
    # Initialize for new logger object;
    if not os.path.exists(LOG_PATH):
        os.mkdir(LOG_PATH)
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    # If no handler present, add one
    if (
        sum([isinstance(handler, RotatingFileHandler) for handler in logger.handlers])
        == 0
    ):
        handler = RotatingFileHandler(
            filename,
            maxBytes=max_log_size,
            backupCount=backup_count,
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)

    return logger


def render_doc_viewer(
    docs_name: str = "documents",
    index_name: str = "current_index",
    include_text: Optional[str] = None,
    unique_key: str = "key",
) -> None:
    """Helper UI function to render a document viewing window.

    Args:
        docs_name (str, optional): st.session_state name holding list of docuements. Defaults to "documents".
        index_name (str, optional): st.session_state name to maintain an object's current index from another. Defaults to "current_index".
        include_text (Optional[str], optional): Optional text to include in the viewer. Defaults to None.
        unique_key (str, optional): Unique key name to maintain multiple document viewers at once. Defaults to "key".
    """
    try:
        docviewer = st.empty()
        idx_placeholder = st.empty()

        # Initialize the current index
        if index_name not in st.session_state:
            st.session_state[index_name] = 0

        # Reset if user filters some docs and current idx is out of bounds
        if st.session_state[index_name] > len(st.session_state[docs_name]) - 1:
            st.session_state[index_name] = 0

        show_next = st.button(
            label=(
                "Next document"
                if not st.session_state.disable_flg
                else "Disabled during process"
            ),
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
                st.write(
                    f"Document: {st.session_state[index_name] + 1} / {len(st.session_state[docs_name])}"
                )

        # Show next element in list
        with docviewer.container(height=300, border=True):
            if include_text:
                st.write(include_text)
            if len(st.session_state[docs_name]) == 0:
                st.write(
                    "No documents with this criteria could be found; try a larger token limit or different search pattern."
                )
            else:
                st.write(st.session_state[docs_name][st.session_state[index_name]])

    except Exception as e:
        logging.error(f"Failed to render document viewer: {e}")


def gather_docs(
    data_processor: data_processors.DataProcessor,
    communicator: communicators.Communicator,
    search_pattern: Optional[str] = None,
    replace_pattern: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """Helper UI function to find documents with specified search pattern and maintain in session states.

    Args:
        data_processor (data_processors.DataProcessor): DataProcessor object with data.
        communicator (communicators.Communicator): Communicator object to count tokens.
        search_pattern (Optional[str], optional): Search string to find matches for in documents. Defaults to None.
        replace_pattern (Optional[str], optional): String to replace search string with. Defaults to None.
        verbose (bool, optional): Write to UI. Defaults to True.
    """
    try:
        if search_pattern is None or replace_pattern is None:
            if verbose:
                st.sidebar.write("Please input search and replace patterns")
        else:
            update_patterns_json(key=search_pattern, val=replace_pattern)
            docs_with_pattern = data_processor.ret_passages_with_pattern(search_pattern)
            if verbose:
                st.sidebar.write(
                    f"{len(docs_with_pattern)} / {len(data_processor.data)} documents with this search pattern."
                )
            docs_valid = [
                p
                for p in docs_with_pattern
                if communicator.count_tokens(p) < config.user_config["TOKEN_LIMIT"]
            ]
            if verbose:
                st.sidebar.write(
                    f"{len(docs_valid)} / {len(docs_with_pattern)} documents under token limit."
                )
            st.session_state["docs_valid"] = docs_valid
            st.session_state["docs_with_pattern"] = docs_with_pattern

    except Exception as e:
        logging.error(f"Failed to gather documents for session states: {e}")


def get_model_factory_name(
    model_name: str = "gpt-3.5-turbo", rag: bool = False
) -> Optional[str]:
    """Helper function mapping vendor API model names to program defined names.

    Args:
        model_name (str, optional): Model name used in vendor API. Defaults to "gpt-3.5-turbo".
        rag (bool, optional): If true, maps model name to RAG version of model. Defaults to False.

    Returns:
        Optional[str]: Program defined model if implemented, else None
    """
    if not rag:
        names_mapped = {
            "gpt-3.5-turbo": "GPT_3.5_TURBO",
            "gpt-4": None,
            "gpt-4-32k": None,
        }
        return names_mapped[model_name]

    else:
        names_mapped = {
            "gpt-3.5-turbo": "GPT_3.5_TURBO_RAG",
            "gpt-4": None,
            "gpt-4-32k": None,
        }
        return names_mapped[model_name]


def btn_lock_callback():
    """Callback function to lock UI widgets upon a triggered event."""
    st.session_state.disable_flg = not st.session_state.disable_flg
