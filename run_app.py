import os
import logging
from logging.handlers import RotatingFileHandler
from config import config
import streamlit as st
import json
from src.communicators import GPTCommunicator
from src.data_processors import WikiTextProcessor
import pandas as pd
from src.helpers import update_patterns_json, process_data, attach_vectorstore

LOG_PATH = config.user_config["LOG_PATH"]
PATTERNS_FILENAME = config.user_config["PATTERNS_FILENAME"]

# Direct src code logging to backend.log through logging.config
logging.basicConfig(
    filename=LOG_PATH+"backend.log",
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)

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
    # if no handler present, add one
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
        button_name="next",
        include_text=None 
    ):
    
    docviewer = st.empty() 
    idx_placeholder = st.empty()

    # Initialize the current index
    if index_name not in st.session_state:
        st.session_state[index_name] = 0

    # reset if user filters some docs and current idx is out of bounds
    if st.session_state[index_name] > len(st.session_state[docs_name]) - 1:
        st.session_state[index_name] = 0

    show_next = st.button(button_name)

    # update index on button click
    if show_next:
        # # loop back to beginning if at end of list
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


@st.cache_data
def init_session():
    # clear logs
    with open(LOG_PATH+"streamlit.log", "w") as _:
        pass
    with open(LOG_PATH+"backend.log", "w") as _:
        pass
    #clear patterns file
    update_patterns_json(clear_json=True)

@st.cache_resource
def init_communicator_and_processor():

    api_key = config.user_config["ACCESS_TOKEN"]
    model_name = config.user_config["MODEL_NAME"]
    verbose = config.user_config["VERBOSE"]

    communicator = GPTCommunicator(api_key=api_key, model_name=model_name)

    data_processor = WikiTextProcessor(
        dataset_version = "wikitext-2-raw-v1", 
        split = "train", 
        communicator = communicator,
        verbose = verbose
    )
    return communicator, data_processor

        
def run_streamlit_app():

    st.set_page_config(page_title="Page")

    # Define divider colors for different sections
    section1_color = "green" # Initializaion section
    section2_color = "blue" # Chatbot section
    sidebar1_color = "orange" # Adjust section
    sidebar2_color = "red" # Log section

    # Init session states and logger
    for state in ["docs_with_pattern", "docs_valid", "enable_doc_viewer", "communicator", "response", "retrieved_docs"]:
        if state not in st.session_state:
            st.session_state[state] = None

    if "logger" not in st.session_state:
        st.session_state["logger"] = create_logger()

    logger = st.session_state["logger"]

    # Objects to be initialized only once on first loop
    init_session()
    communicator, data_processor = init_communicator_and_processor()

    st.title("Title")

    # Initialization sections of UI

    st.header("Initialization", divider=section1_color)

    ## Section to show current param vals

    st.subheader("Config Params", divider=section1_color)

    ### Init with default config vals
    token_count_placeholder = st.empty()
    with token_count_placeholder.container(border=False):
        st.write(f"Token limit: {config.user_config['TOKEN_LIMIT']}")

    model_placeholder = st.empty()
    with model_placeholder.container(border=False):
        st.write(f"Model: {config.user_config['MODEL_NAME']}")

    ndocs_placeholder = st.empty()
    with ndocs_placeholder.container(border=False):
        st.write(f"Number of docs to retrieve: {config.user_config['N_RETRIEVED_DOCS']}")

    ## Section to document manipulation through Streamlit UI
        
    st.sidebar.header("Manipulate Documents", divider=sidebar1_color)

    search_pattern_col, replace_pattern_col = st.sidebar.columns(2)
    search_pattern = search_pattern_col.text_input("Search Pattern")
    replace_pattern = replace_pattern_col.text_input("Replace Pattern")

    if st.sidebar.button("Add Pattern"):
        update_patterns_json(key=search_pattern, val=replace_pattern)
        docs_with_pattern = data_processor.ret_passages_with_pattern(search_pattern)
        st.sidebar.write(f"{len(docs_with_pattern)} / {len(data_processor.data)} documents with this search pattern.")
        docs_valid = [p for p in docs_with_pattern if communicator.count_tokens(p) < config.user_config["TOKEN_LIMIT"]]
        st.sidebar.write(f"{len(docs_valid)} / {len(docs_with_pattern)} documents under token limit.")
        st.session_state["docs_valid"] = docs_valid
        st.session_state["docs_with_pattern"] = docs_with_pattern
        
    if st.session_state["docs_valid"]:
        if st.sidebar.button("View these documents"):
            st.session_state["enable_doc_viewer"] = True

    if st.session_state["enable_doc_viewer"]:
        st.subheader("Doc Viewer", divider=section1_color)

        if st.session_state["docs_valid"] != None:
            # if st.checkbox("Filter to documents under token limit"):
            #     # reset index to avoid potential out of bounds error
            #     st.session_state["search_match_index"] = 0
            #     docs_used = [p for p in st.session_state["docs_valid"] if communicator.count_tokens(p) <= config.user_config["TOKEN_LIMIT"]]
            #     st.session_state["docs_valid"] = docs_used 

            n_filtered = len(st.session_state["docs_with_pattern"]) - len(st.session_state["docs_valid"])

            st.write(f"{n_filtered} / {len(st.session_state['docs_with_pattern'])} filtered out due to token limit")

            render_doc_viewer(
                docs_name="docs_valid", 
                index_name="search_match_index",
                button_name="Next search match",
                include_text=f"Search: {search_pattern}",
            )

    if st.sidebar.button("Clear Patterns"):
        update_patterns_json(clear_json=True)
        st.session_state["docs_valid"] = None

    with open(PATTERNS_FILENAME) as r:
        patterns_json = json.load(r)

    data = {"search": patterns_json.keys(), "replace": patterns_json.values()}

    st.subheader("Document Manipulation", divider=section1_color)

    st.write(pd.DataFrame(data))

    ## Sidebar section to adjust params

    ### Input to change model
    selected_model = st.sidebar.selectbox(
        'Select a model:',
        (
            "gpt-3.5-turbo", 
            "gpt-4", 
            "gpt-4-32k"
        )
    )
    if selected_model:
        config.user_config["MODEL_NAME"] = selected_model
        with model_placeholder.container(border=False):
            st.write(f"Model: {selected_model}")

    ### Input to adjust token limit
    token_limit = st.sidebar.text_input("Token Limit")
    
    if token_limit:
        token_limit = int(token_limit)
        config.user_config["TOKEN_LIMIT"] = token_limit
        with token_count_placeholder.container(border=False):
            st.write(f"Token limit: {token_limit}")

    ### Input to adjust n retrieved docs     
    ndocs = st.sidebar.text_input("Number of docs to retrieve")
    
    if ndocs:
        ndocs = int(ndocs)
        config.user_config["N_RETRIEVED_DOCS"] = ndocs
        with ndocs_placeholder.container(border=False):
            st.write(f"Number of docs to retrieve: {ndocs}")

    # Chatbot interaction section

    ## Vectorstore init with manipulated docs
    st.header("Chatbot QA", divider=section2_color)

    load_vs, create_vs = st.columns(2)

    if load_vs.button("Load vectorstore"):
        process_data(data_processor)
        communicator = attach_vectorstore(communicator, load_vectorstore=True)
        st.session_state["communicator"] = communicator

    elif create_vs.button("Create new vectorstore"):
        process_data(data_processor)
        result_holder = st.empty()

        def progress(p, i):
            with result_holder.container():
                st.progress(p, f'Progress: Documents Processed={i}')

        communicator = attach_vectorstore(communicator, load_vectorstore=False, _callback=progress)
        st.session_state["communicator"] = communicator

    else:
        pass

    ## Send/receive with GPT
    if st.session_state["communicator"] != None:
        user_query = st.text_input("Query: ")

        if st.button("Get Answer"):
            if user_query:

                # Perform RAG based on user query
                response, retrieved_context = st.session_state["communicator"].post_rag_prompt(user_query)
                
                st.session_state["response"] = response
                st.session_state["retrieved_docs"] = retrieved_context

                logger.info(f"USER QUERY: {user_query}")
                logger.info(f"GPT RESPONSE: {response}")

    if st.session_state["response"] != None:
        st.subheader("GPT Response", divider=section2_color)
        response_placeholder = st.empty()
        with response_placeholder.container(border=True):
            st.write(st.session_state["response"])

    if st.session_state["retrieved_docs"] != None:

        st.subheader("Retrieved Context", divider=section2_color)

        if st.checkbox("Show"):
            render_doc_viewer(
                docs_name="retrieved_docs", 
                index_name="context_index",
                button_name="Next retrieved document",
                #include_text=f"Retrieved documents for query",
            )

    # Logging section

    # Capture messages from this session
    st.sidebar.header("Session logs", divider=sidebar2_color)
    with open(LOG_PATH+"streamlit.log") as log:
        st.sidebar.write(log.readlines())     

    # Backend log
    st.sidebar.header("Developer logs", divider=sidebar2_color)
    with open(LOG_PATH+"backend.log") as log:
        st.sidebar.write(log.readlines())

run_streamlit_app()
