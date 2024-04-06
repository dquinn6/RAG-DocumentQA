"""Module containing factory constructors for other modules in this codebase."""

import json
from typing import Callable, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings

from src.communicators import Communicator, GPTCommunicator
from src.config import config
from src.data_processors import DataProcessor, WikiTextProcessor
from src.utils import test_communication
from src.vectorstore_handlers import LangchainVectorstoreFAISS

API_KEY = config.user_config["ACCESS_TOKEN"]
MODEL_NAME = config.user_config["MODEL_NAME"]
SAVE_PATH = config.user_config["SAVE_PATH"]
SEARCH_TYPE = config.user_config["SEARCH_TYPE"]
N_RETRIEVED_DOCS = config.user_config["N_RETRIEVED_DOCS"]
VERBOSE = config.user_config["VERBOSE"]
PATTERNS_FILENAME = config.user_config["PATTERNS_FILENAME"]


class DataProcessorFactory:
    """Constructor for data processing objects."""

    implemented_classes = ["WikiText"]

    def create_processor(
        self,
        name: str = "WikiText",
        communicator: Optional[Communicator] = None,
        process_data: bool = True,
    ) -> DataProcessor:
        """Construct the data processor object based on the given name.

        Args:
            name (str): Type of data processor to construct; must be an implemented class. Defaults to "WikiText".
            communicator (Optional[Communicator], optional): Optional Communicator for token limiting. Defaults to None.
            process_data (bool, optional): If true, process the initialized data. Defaults to True.

        Raises:
            NotImplementedError: Raised when given a name for a class that hasn't been implemented.

        Returns:
            DataProcessor: Object for data processing.
        """
        # Define class based on given name
        if name == "WikiText":
            processor = WikiTextProcessor(
                dataset_version="wikitext-2-raw-v1",
                split="train",
                communicator=communicator,
                verbose=VERBOSE,
            )
        else:
            raise NotImplementedError(
                f"'{name}' DataProcessor not implemented; valid names include: {self.implemented_classes}"
            )

        if process_data:
            with open(PATTERNS_FILENAME) as f:
                patterns = list(json.load(f).items())

            _ = processor.process_text(
                token_limit=config.user_config["TOKEN_LIMIT"],
                save_path=SAVE_PATH,
                save_filename="processed_data.csv",
                manipulate_pattern=patterns,
            )

        return processor


class VectorstoreFactory:
    """Constructor for vectorstore handler objects."""

    implemented_classes = ["LangchainFAISS"]

    def attach_vectorstore(
        self,
        communicator: Communicator,
        name: str = "LangchainFAISS",
        load_vectorstore: bool = True,
        _callback: Optional[Callable] = None,
    ) -> None:
        """Attach a vectorstore to a given Communicator object to enable RAG querying.

        Args:
            communicator (Communicator): Communicator object to enable RAG for.
            name (str, optional): Type of vectorstore to use. Defaults to "LangchainFAISS".
            load_vectorstore (bool, optional): Whether to load vectorstore currently saved vectorstore or create new one. Defaults to True.
            _callback (Optional[Callable], optional): Optional callback function. Defaults to None.

        Raises:
            NotImplementedError: Raised when given a name for a class that hasn't been implemented.

        """
        # Define class based on given name
        if name == "LangchainFAISS":
            vs = LangchainVectorstoreFAISS(
                embedding_type=HuggingFaceEmbeddings(),
                processed_csv_path=SAVE_PATH + "processed_data.csv",
                verbose=VERBOSE,
            )
            if load_vectorstore:
                vs.load_local_vectorstore(load_path=SAVE_PATH)
            else:
                vs.create_local_vectorstore(
                    save_path=SAVE_PATH, force_create=True, callback=_callback
                )

            # Create document retriever with given args
            vs.create_retriever(
                search_type=SEARCH_TYPE, search_kwargs={"k": N_RETRIEVED_DOCS}
            )
            # Attach vectorstore handler to given Communicator
            communicator.set_vectorstore_handler(vs)

        else:
            raise NotImplementedError(
                f"'{name}' Vectorstore not implemented; valid names include: {self.implemented_classes}"
            )


class ModelFactory:
    """Constructor for model communicator objects."""

    implemented_classes = ["GPT_3.5_TURBO", "GPT_3.5_TURBO_RAG", "GPT_4", "GPT_4_RAG"]

    def create_model(
        self,
        model_name: str = "GPT_3.5_TURBO",
        dataset_name: str = "WikiText",
        vectorstore_name: str = "LangchainFAISS",
        new_vectorstore: bool = False,
        _callback: Optional[Callable] = None,
    ) -> Communicator:
        """Construct model communicator, with RAG variants if specified.

        Args:
            model_name (str, optional): Type of model to construct, with "_RAG" appendage for RAG variants. Defaults to "GPT_3.5_TURBO".
            dataset_name (str, optional): DataProcessor type of optional new vectorstore creation. Defaults to "WikiText".
            vectorstore_name (str, optional): Vectorstore type for RAG variants. Defaults to "LangchainFAISS".
            new_vectorstore (bool, optional): Whether to create a new vectorstore or load currently saved. Defaults to False.
            _callback (Optional[Callable], optional): Optional callback function. Defaults to None.

        Raises:
            NotImplementedError: Raised when given a name for a class that hasn't been implemented.

        Returns:
            Communicator: Model communicator object.
        """

        def transform_to_rag_model(
            communicator, dataset_name, vectorstore_name, new_vectorstore
        ):
            """Attachs a vectorstore to a communicator object."""
            # Create Dataset processor and process dataset with config.yml params
            if (
                new_vectorstore
            ):  # only need to reprocess data for new vectorstore creation
                data_factory = DataProcessorFactory()
                _ = data_factory.create_processor(
                    dataset_name, communicator, process_data=True
                )

            # Create vectorstore and attach to communicator to create RAG model
            vs_factory = VectorstoreFactory()
            vs_factory.attach_vectorstore(
                communicator,
                vectorstore_name,
                load_vectorstore=not new_vectorstore,
                _callback=_callback,
            )

        if model_name not in self.implemented_classes:
            raise NotImplementedError(
                f"'{model_name}' Model not implemented; valid names include: {self.implemented_classes}"
            )

        # Dictionary mapping factory names to model names e.g. GPT_3.5_TURBO_RAG will still be using API model gpt-3.5-turbo
        factory_name_to_api_name = {
            "GPT_3.5_TURBO": "gpt-3.5-turbo",
            "GPT_3.5_TURBO_RAG": "gpt-3.5-turbo",
            "GPT_4": "gpt-4",
            "GPT_4_RAG": "gpt-4",
        }

        # Init communicator with API model name
        api_model_name = factory_name_to_api_name[model_name]
        communicator = GPTCommunicator(api_key=API_KEY, model_name=api_model_name)

        # If RAG version chosen, transform the model
        if model_name.split("_")[-1] == "RAG":
            transform_to_rag_model(
                communicator, dataset_name, vectorstore_name, new_vectorstore
            )

        # Test communication for potential issues (e.g. bad API key, server down, etc.)
        test_communication(communicator)

        return communicator
