# Retrieval Augmented Generation (RAG) Application Demo

## About
This application is a Retrieval Augmented Generation (RAG) based system using a Large Language Model (LLM) to answer questions on a supported set of documents. This demo application currently uses [OpenAI's GPT](https://platform.openai.com/docs/models) as the underlying LLM and the [WikiText2](https://huggingface.co/datasets/wikitext) dataset as a set of dummy documents to answer questions on, but the modular structure of this codebase allows a developer to easily implement other types of models or datasets.

For a real use case, we would normally supply a set of private documents to the LLM that it wasn't trained on, creating a much more practical question answering application. WikiText2 is used in this application as it is open-source and provides a diverse set of documents to query on. However, GPT has already been trained on much of the content in these documents and can answer related questions without using a RAG approach. To better demonstrate this system, the functionality to manipulate these documents has been incorporated to better demonstrate GPT's usage of the given information rather than its internal knowledge.

## Running the UI-based application

The program has a UI implemented with Streamlit for easily manipulating the documents, creating the vectorstore, and interacting with the RAG-enabled model. To start the app, navigate to the root project directory and run the following command:

    streamlit run run_app.py

The loaded page has a 'usage' section that describes how to use the interface; you can also see a demo usage video in the UI Demo section below. 

The default configuration will run the program in your local browser, but streamlit can be configured to be deployed on a remote server for user servicing if desired. 

### Stopping the program

To stop the program, send the terminate process signal (CTRL+C) in your terminal. You can also suspend the program (CTRL+Z) and terminate the program with the command 'killall -9 python'.

### UI Demo

![](readme_images/Demo.gif)


## Running the CLI-based application

If you don't wish to use the UI-based app, you can interface with the base RAG system using the command line interface via run_cli.py. This program will create the model using the config params and manipulation patterns in src/config/user_config.yml and src/config/manipulate_patterns.json. After setting your desired params for the run, use the following command to start the program:

    python run_cli.py

### CLI Demo

![](readme_images/Demo_CLI.png)
