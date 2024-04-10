"""Application script for WikiText RAG QA demo through Streamlit UI."""

import json
import logging
from importlib import reload

import pandas as pd
import streamlit as st

from src.app_helpers import (
    btn_lock_callback,
    create_logger,
    create_placeholder_files,
    gather_docs,
    get_model_factory_name,
    render_doc_viewer,
)
from src.config import config
from src import factories
#from src.factories import DataProcessorFactory, ModelFactory
from src.utils import update_config_yml, update_patterns_json

LOG_PATH = config.user_config["LOG_PATH"]
PATTERNS_FILENAME = config.user_config["PATTERNS_FILENAME"]

# Define dataset, vectorstore, and model in config.yml to use in this app
DATASET_NAME = config.user_config["DATASET_NAME"]  # WikiText
VECTORSTORE_NAME = config.user_config["VECTORSTORE_NAME"]  # FAISS
MODEL_NAME = config.user_config[
    "MODEL_NAME"
]  # name that matches openai names; needs to be mapped to defined factory names

# Create empty logfiles, pattern files, and directories if they don't exist
create_placeholder_files()

# Direct src code logging to backend.log
logging.basicConfig(
    filename=LOG_PATH + "backend.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


@st.cache_data
def init_files():
    """Clear log files and patterns json on new session start-up."""
    with open(LOG_PATH + "streamlit.log", "w") as f:
        f.close()
    with open(LOG_PATH + "backend.log", "w") as f:
        f.close()
    update_patterns_json(clear_json=True)


@st.cache_resource
def init_communicator_and_processor():
    """Init communicator and data processor on start-up."""
    # Since user will manipulate docs from UI, init with regular communicator first
    model_factory = factories.ModelFactory()
    communicator = model_factory.create_model(
        get_model_factory_name(MODEL_NAME, rag=False)
    )

    data_factory = factories.DataProcessorFactory()
    data_processor = data_factory.create_processor(DATASET_NAME, communicator)

    return communicator, data_processor


def run_streamlit_app():
    """Perform main UI logic; Streamlit runs this loop continuously."""
    st.set_page_config(page_title="App", layout="wide")

    # Define divider colors for different sections
    section1_color = "green"  # Initializaion section
    section2_color = "blue"  # Chatbot section
    sidebar1_color = "orange"  # Adjust section
    sidebar2_color = "red"  # Log section

    # Init session states and logger
    for state in [
        "docs_with_pattern",
        "docs_valid",
        "enable_doc_viewer",
        "communicator",
        "response",
        "retrieved_docs",
    ]:
        if state not in st.session_state:
            st.session_state[state] = None

    if "disable_flg" not in st.session_state:
        st.session_state["disable_flg"] = False

    if "logger" not in st.session_state:
        st.session_state["logger"] = (
            create_logger()
        )  # Direct UI logging to streamlit.log

    logger = st.session_state["logger"]

    # Objects to be initialized only once on first loop
    init_files()
    communicator, data_processor = init_communicator_and_processor()

    # About and usage section

    st.title("Manipulate WikiText2 QA")
    hide_preface = st.checkbox("Hide Preface", key="hide_preface")

    about_text = "This application is a RAG-based system using GPT to answer questions on the WikiText2 dataset, \
    which is used as a set of dummy documents for this demo. Since GPT has already been trained on content \
    in these documents and can answer related questions without RAG, the functionality to manipulate them has \
    been incorporated to better demonstrate GPT's usage of the given information over its internal knowledge."

    if not hide_preface:
        st.header("About")
        st.write(about_text)

    usage_text = "The Initialization section below lists the current configuration for this session. Use the input sections\
    in the left sidebar to adjust these parameters before creating the system. Here you will also be able to manipulate the documents \
    that GPT will be using to answer questions. When you provide search and replace text and add the pattern, a viewer will appear to show\
    all documents that have this search string. You can use this viewer to read the docs and create questions GPT should be able to answer.\
    You can add as many search and replace patterns as you like; the Document Manipulation section will keep track of every added pattern.\
    \n\
    \nOnce the configuration is done and document manipulation has been set, move down to the Chatbot QA section and click the 'Create vectorstore' button to build a vectorstore database\
    with the manipulated documents. While this is processing, you won't be able to interact with other parts of the UI until the vectostore is created.\
    Alternatively, you can click the 'Load vectorstore' button to utilize the vectorstore from the previous session and avoid rebuilding.\
    \n\
    \nAfter the vectorstore is set, you'll be able to interact with GPT by sending queries and viewing responses based off of the manipulated documents!"

    if not hide_preface:
        st.header("Usage")
        st.write(usage_text)

    # Initialization sections of UI

    st.header("Initialization", divider=section1_color)

    st.write(
        f"This section ({section1_color}) shows the current configuration that will be used when creating the vectorstore."
    )

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
        st.write(
            f"Number of docs to retrieve: {config.user_config['N_RETRIEVED_DOCS']}"
        )

    ## Sidebar section to adjust params
    st.sidebar.header("Choose model", divider=sidebar1_color)

    ### Input to change model
    selected_model = st.sidebar.selectbox(
        "Select a model:",
        (
            "gpt-3.5-turbo",
            "gpt-4",
        ),
        disabled=st.session_state.disable_flg,
    )
    if selected_model:
        update_config_yml({"MODEL_NAME": selected_model})
        with model_placeholder.container(border=False):
            st.write(f"Model: {selected_model}")

    ## Section for document manipulation through Streamlit UI

    st.sidebar.header("Manipulate Documents", divider=sidebar1_color)

    ### Capture user input for search and replace
    search_pattern_col, replace_pattern_col = st.sidebar.columns(2)
    search_pattern = search_pattern_col.text_input(
        "Search Pattern", disabled=st.session_state.disable_flg
    )
    replace_pattern = replace_pattern_col.text_input(
        "Replace Pattern", disabled=st.session_state.disable_flg
    )

    add_button_col, clear_button_col = st.sidebar.columns([1, 1])

    with add_button_col:
        if st.sidebar.button(
            label=(
                "Add Pattern"
                if not st.session_state.disable_flg
                else "Disabled during process"
            ),
            disabled=st.session_state.disable_flg,
            key="add",
        ):
            gather_docs(
                data_processor,
                communicator,
                search_pattern,
                replace_pattern,
                verbose=True,
            )
            st.session_state["enable_doc_viewer"] = True

    with clear_button_col:
        if st.sidebar.button(
            (
                "Clear Patterns"
                if not st.session_state.disable_flg
                else "Disabled during process"
            ),
            disabled=st.session_state.disable_flg,
            key="clear",
        ):
            update_patterns_json(clear_json=True)
            st.session_state["docs_valid"] = None
            st.session_state["enable_doc_viewer"] = False

    ### Input to adjust token limit
    token_limit = st.sidebar.text_input(
        "Token Limit", disabled=st.session_state.disable_flg
    )

    # Update docs if user input a token limit and it's different from config val
    if token_limit:
        token_limit = int(token_limit)
        if config.user_config["TOKEN_LIMIT"] != token_limit:
            # config.user_config["TOKEN_LIMIT"] = token_limit
            update_config_yml({"TOKEN_LIMIT": token_limit})
            with token_count_placeholder.container(border=False):
                st.write(f"Token limit: {token_limit}")
            # Update docs with new token limit
            gather_docs(
                data_processor,
                communicator,
                search_pattern,
                replace_pattern,
                verbose=False,
            )

    ### Input to adjust n retrieved docs
    ndocs = st.sidebar.text_input(
        "Number of docs to retrieve", disabled=st.session_state.disable_flg
    )

    if ndocs:
        ndocs = int(ndocs)
        update_config_yml({"N_RETRIEVED_DOCS": ndocs})
        with ndocs_placeholder.container(border=False):
            st.write(f"Number of docs to retrieve: {ndocs}")

    if st.session_state["enable_doc_viewer"]:
        st.subheader("Doc Viewer", divider=section1_color)

        if st.session_state["docs_valid"] is not None:
            n_filtered = len(st.session_state["docs_with_pattern"]) - len(
                st.session_state["docs_valid"]
            )

            st.write(
                f"{n_filtered} / {len(st.session_state['docs_with_pattern'])} filtered out due to token limit"
            )

            render_doc_viewer(
                docs_name="docs_valid",
                index_name="search_match_index",
                include_text=f"Search: {search_pattern}",
                unique_key="stored_documents",
            )

    with open(PATTERNS_FILENAME) as r:
        patterns_json = json.load(r)

    data = {"search": patterns_json.keys(), "replace": patterns_json.values()}

    st.subheader("Document Manipulation", divider=section1_color)

    st.write(pd.DataFrame(data))

    # Chatbot interaction section

    st.header("Chatbot QA", divider=section2_color)

    ## Vectorstore creation with user defined params from UI input

    load_vs, create_vs = st.columns(2)

    if load_vs.button("Load vectorstore", key="load"):
        try:
            # Need to reload factories module, since config may have been adjusted and differ from the current module variables
            reload(factories)

            model_factory = factories.ModelFactory()
            communicator = model_factory.create_model(
                model_name=get_model_factory_name(MODEL_NAME, rag=True),
                dataset_name=DATASET_NAME,
                vectorstore_name=VECTORSTORE_NAME,
                new_vectorstore=False,
            )
            st.session_state["communicator"] = communicator

        except Exception as e:
            st.write(
                "Failed to load current vectorstore; please create a new one to proceed."
            )
            error_placeholder = st.empty()
            with error_placeholder.container(border=True):
                st.write(e)

    if create_vs.button(
        "Create new vectorstore", on_click=btn_lock_callback, key="create"
    ):
        # Disable user input during process to avoid breaking
        with st.spinner("Wait for process to finish..."):
            st.session_state["disable_flg"] = True
            progress_holder = st.empty()

            def progress(p, i):
                with progress_holder.container():
                    st.progress(p, f"Progress: Documents Processed={i}")

            # Need to reload factories module, since config may have been adjusted and differ from the current module variables
            reload(factories)

            model_factory = factories.ModelFactory()

            communicator = model_factory.create_model(
                model_name=get_model_factory_name(MODEL_NAME, rag=True),
                dataset_name=DATASET_NAME,
                vectorstore_name=VECTORSTORE_NAME,
                new_vectorstore=True,
                _callback=progress,
            )
            st.session_state["communicator"] = communicator

    ## Send/receive with GPT
    if st.session_state["communicator"] is not None:
        # Enable buttons and refresh
        if st.session_state["disable_flg"]:
            st.session_state["disable_flg"] = False
            st.rerun()

        user_query = st.text_input("Query: ")

        if st.button("Get Answer", key="answer"):
            if user_query:

                # Perform RAG based on user query
                response, retrieved_context = st.session_state[
                    "communicator"
                ].post_rag_prompt(user_query)

                st.session_state["response"] = response
                st.session_state["retrieved_docs"] = retrieved_context

                logger.info(f"USER QUERY: {user_query}")
                logger.info(f"GPT RESPONSE: {response}")

    if st.session_state["response"] is not None:
        st.subheader("GPT Response", divider=section2_color)
        response_placeholder = st.empty()
        with response_placeholder.container(border=True):
            st.write(st.session_state["response"])

    if st.session_state["retrieved_docs"] is not None:

        st.subheader("Retrieved Context", divider=section2_color)

        if st.checkbox("show", key="show_retrieved"):
            render_doc_viewer(
                docs_name="retrieved_docs",
                index_name="context_index",
                unique_key="retrieved_documents",
            )

    # Logging section

    # Capture messages from this session
    st.sidebar.header("Session logs", divider=sidebar2_color)
    with open(LOG_PATH + "streamlit.log") as log:
        st.sidebar.write(log.readlines())

    # Backend log
    st.sidebar.header("Developer logs", divider=sidebar2_color)
    with open(LOG_PATH + "backend.log") as log:
        st.sidebar.write(log.readlines())


run_streamlit_app()
