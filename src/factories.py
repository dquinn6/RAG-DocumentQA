from src.config import config
from src.communicators import GPTCommunicator
from src.data_processors import WikiTextProcessor
from src.vectorstore_handlers import LangchainVectorstore
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from src.utils import test_communication

API_KEY = config.user_config["ACCESS_TOKEN"]
MODEL_NAME = config.user_config["MODEL_NAME"]
SAVE_PATH = config.user_config["SAVE_PATH"]
SEARCH_TYPE = config.user_config["SEARCH_TYPE"]
N_RETRIEVED_DOCS = config.user_config["N_RETRIEVED_DOCS"]
#TOKEN_LIMIT = config.user_config["TOKEN_LIMIT"]
VERBOSE = config.user_config["VERBOSE"]
PATTERNS_FILENAME = config.user_config["PATTERNS_FILENAME"]

    
class DataProcessorFactory:

    implemented_classes = ["WikiText"]

    def create_processor(self, name, communicator=None, process_data=True):

        if name == "WikiText":
            processor = WikiTextProcessor(
                dataset_version = "wikitext-2-raw-v1", 
                split = "train", 
                communicator = communicator,
                verbose = VERBOSE
            )
        else:
            raise NotImplementedError(f"'{name}' DataProcessor not implemented; valid names include: {self.implemented_classes}")
        
        if process_data:
            with open(PATTERNS_FILENAME) as f:
                patterns = list(json.load(f).items())

            _ = processor.process_text(
                token_limit = config.user_config["TOKEN_LIMIT"], 
                save_path = SAVE_PATH,
                save_filename = "processed_data.csv",
                manipulate_pattern = patterns
            )

        return processor
        
    
class VectorstoreFactory:

    implemented_classes = ["Langchain"]

    def attach_vectorstore(self, name, communicator, load_vectorstore=True, _callback=None):

        if name == "Langchain":
            vs = LangchainVectorstore(
                embedding_type = HuggingFaceEmbeddings(),
                processed_csv_path = SAVE_PATH+"processed_data.csv",
                verbose = VERBOSE
            )
            if load_vectorstore:
                vs.load_local_vectorstore(load_path=SAVE_PATH)
            else:
                vs.create_local_vectorstore(save_path=SAVE_PATH, force_create=True, callback=_callback)

            vs.create_retriever(
                search_type=SEARCH_TYPE,
                search_kwargs={
                    "k": N_RETRIEVED_DOCS
                }
            )

            communicator.set_vectorstore_handler(vs)

        else:
            raise NotImplementedError(f"'{name}' Vectorstore not implemented; valid names include: {self.implemented_classes}")

        
class ModelFactory:

    implemented_classes = ["GPT_3.5_TURBO", "GPT_3.5_TURBO_RAG", "GPT_4", "GPT_4_RAG"]

    def create_model(self, model_name, dataset_name="WikiText", vectorstore_name="Langchain", new_vectorstore=False, _callback=None):

        def transform_to_rag_model(communicator, dataset_name, vectorstore_name, new_vectorstore):

            # Create Dataset processor and process dataset with config.yml params
            if new_vectorstore: # only need to reprocess data for new vectorstore creation
                data_factory = DataProcessorFactory()
                _ = data_factory.create_processor(dataset_name, communicator, process_data=True)

            # Create vectorstore and attach to communicator to create RAG model
            vs_factory = VectorstoreFactory()
            vs_factory.attach_vectorstore(vectorstore_name, communicator, load_vectorstore=not new_vectorstore, _callback=_callback)

        if model_name not in self.implemented_classes:
            raise NotImplementedError(f"'{model_name}' Model not implemented; valid names include: {self.implemented_classes}")

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
            transform_to_rag_model(communicator, dataset_name, vectorstore_name, new_vectorstore)

        # Test communication for potential issues (e.g. bad API key, server down, etc.)
        test_communication(communicator)
            
        return communicator