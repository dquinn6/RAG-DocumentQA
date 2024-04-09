# Retrieval Augmented Generation (RAG) Application Demo

## Table of Contents

[About](#bout)

[Getting Started](#getting-started)

- [Setup](#setup)
- [Documentation](#documentation)

[Running the UI-based Application](#running-the-ui-based-application)

- [UI Demo](#ui-demo)
- [Stopping the program](#stopping-the-program)

[Running the CLI-based Application](#running-the-cli-based-application)

- [CLI Demo](#cli-demo)

<br/>

## About
This application is a Retrieval Augmented Generation (RAG) based system using a Large Language Model (LLM) to answer questions on a supported set of documents. This demo application currently uses [OpenAI's GPT](https://platform.openai.com/docs/models) as the underlying LLM and the [WikiText2](https://huggingface.co/datasets/wikitext) dataset as a set of dummy documents to answer questions on, but the modular structure of this codebase allows a developer to easily implement other types of models or datasets.

For a real use case, we would normally supply a set of private documents to the LLM that it wasn't trained on, creating a much more practical question answering application. WikiText2 is used in this application as it is open-source and provides a diverse set of documents to query on. However, GPT has already been trained on much of the content in these documents and can answer related questions without using a RAG approach. To better demonstrate this system, the functionality to manipulate these documents has been incorporated to better demonstrate GPT's usage of the given information rather than its internal knowledge.

<br/>

## Getting Started

### Setup

#### 1. Install Anaconda

Installing Anaconda is the easiest way to get Python, environments, and package managers set up on your device, providing everything you need to run this program. If not already installed, navigate to the [Anaconda download page](https://www.anaconda.com/download) and install on your device before proceeding.

#### 2. Create a new Python environment

    conda create -n YOUR_ENV_NAME python=3.11
    conda activate YOUR_ENV_NAME

#### 3. Install this project as a package

    cd PROJECT_DIR
    pip install .

#### 4. Initialize the codebase with required files.

Run the following command to create placeholder files required for the application.

    python run_init.py

#### 5. Set your model API key

Navigate to src/config/user_config.yml and add your OpenAI API key under ACCESS_TOKEN.

#### 6. (Optional) Change other config parameters.

Change other parameters in user_config.yml if desired. However, many of these can be adjusted from the UI after starting the program. See [Config](#config) for more details on each of these parameters.


<br/>

## Running the UI-based Application

The program has a UI implemented with Streamlit for easily manipulating the documents, creating the vectorstore, and interacting with the RAG-enabled model. To start the app, navigate to the root project directory and run the following command:

    streamlit run run_app.py

The loaded page has a 'usage' section that describes how to use the interface; you can also see a demo usage video in the UI Demo section below. 

The default configuration will run the program in your local browser, but streamlit can be configured to be deployed on a remote server for user servicing if desired. 

### UI Demo

![](readme_images/Demo.gif)

### Stopping the program

To stop the program, send the terminate process signal (CTRL+C) in your terminal. You can also suspend the program (CTRL+Z) and terminate the program with the command 'killall -9 python'.

<br/>

## Running the CLI-based Application

If you don't wish to use the UI-based app, you can interface with the base RAG system using the command line interface via run_cli.py. This program will create the model using the config params and manipulation patterns in src/config/user_config.yml and src/config/manipulate_patterns.json. After setting your desired params for the run, use the following command to start the program:

    python run_cli.py

### CLI Demo

![](readme_images/Demo_CLI.png)

<br/>

## Config

**ACCESS_TOKEN**: OpenAI API key for model access; may be expanded for other types of models in the future.

**MODEL_NAME**: Type of GPT model to use e.g. gpt-3.5-turbo.


**DATASET_NAME**: Implemented dataset name to use; currently only supports: ["WikiText"].

**VECTORSTORE_NAME**: Implemented vectorstore name to use; currently only supports: ["LangchainFAISS"].

**SEARCH_TYPE**: Type of embedding search algorithm to use for vectorstore lookup; currently only supports: ["similarity", "mmr", "similarity_score_threshold"].

**TOKEN_LIMIT**: Max number (int) of tokens to limit documents to.

**N_RETRIEVED_DOCS**: Number of top matching documents from retrieval to provide to model.

**SAVE_PATH**: Path to save run artifacts to.


**LOG_PATH**: Path to save program logs to.

**PATTERNS_FILENAME**: Path+filename to json with search:replace patterns to manipulate documents with.

**VERBOSE**: Show logging INFO messages.

<br/>

## Documentation

    cd PROJECT_DIR
    pdoc --html src

    cd PROJECT_DIR
    start docs/src/index.html
