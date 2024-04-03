import pytest
from src.communicators import GPTCommunicator
from src.data_processors import WikiTextProcessor
from src.vectorstore_handlers import LangchainVectorstore
from src.factories import DataProcessorFactory, VectorstoreFactory, ModelFactory
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.utils import update_config_yml
from config import config
import logging
import json
from src.utils import update_patterns_json
import string
import random

API_KEY = config.user_config["ACCESS_TOKEN"]
MODEL_NAME = config.user_config["MODEL_NAME"]
SAVE_PATH = config.user_config["SAVE_PATH"]
SEARCH_TYPE = config.user_config["SEARCH_TYPE"]
N_RETRIEVED_DOCS = config.user_config["N_RETRIEVED_DOCS"]
TOKEN_LIMIT = config.user_config["TOKEN_LIMIT"]
VERBOSE = config.user_config["VERBOSE"]

PATTERNS_FILENAME = config.user_config["PATTERNS_FILENAME"]

class TestGPT:

    example_string = "This is a sentence. This is a longer sentence. This is an even longer sentence."

    @pytest.fixture
    def gpt(self):
        return GPTCommunicator(api_key=API_KEY, model_name=MODEL_NAME)

    def test_communication(self, gpt):
        response = gpt.post_prompt("Hi")
        assert isinstance(response, str), "Bad response"

    def test_raises_exception_on_rag_without_vs(self, gpt):
        with pytest.raises(ValueError):
            gpt.post_rag_prompt("Hi")
            
    def test_raises_exception_on_bad_name(self):
        with pytest.raises(ValueError):
            _ = GPTCommunicator(api_key=API_KEY, model_name="bad_name")

    def test_token_count(self, gpt):
        expected_count = 18
        token_count_example = gpt.count_tokens(self.example_string)
        assert token_count_example == expected_count, f"Failed token count; counted {token_count_example} but expected {expected_count}"

    def test_truncate(self, gpt):
        token_limit = 10
        truncated = gpt.truncate_text(self.example_string, token_limit)
        truncated_count = gpt.count_tokens(truncated)
        assert truncated_count <= token_limit, f"Failed truncate; counted {truncated_count} after truncation with token_limit {token_limit}"

    
class TestWikiTextProcessor:

    example_string = "New York State Route"

    @pytest.fixture
    def processor(self):
        communicator = GPTCommunicator(api_key=API_KEY, model_name=MODEL_NAME)
        processor = WikiTextProcessor(
            dataset_version = "wikitext-2-raw-v1", 
            split = "train", 
            communicator = communicator,
            verbose = False
        )
        return processor

    @pytest.mark.parametrize("string_with_delimiter,expected", [
        (" = Title = \n", "title"),
        (" = = Header = = \n", "header"),
        (" = = = Subheader = = = \n", "subheader"),
        ("Content \n", "content")
    ])
    def test_string_classify(self, processor, string_with_delimiter, expected):
        assert processor.classify_string_type(string_with_delimiter) == expected, "Bad string classification"

    def test_pattern_match(self, processor):
        passage_matches = processor.ret_passages_with_pattern(self.example_string)
        assert len(passage_matches) > 0, "Bad test; no passages with this example_string found"
        assert all([self.example_string in passage for passage in passage_matches])

    def test_passage_manipulation(self, processor):

        replace_with = "Some other string"

        passages = processor.process_text(
            token_limit = 500, 
            save_path = None,
            manipulate_pattern = [(self.example_string, replace_with)],
        )
        assert len(passages) > 0, "Bad search; no passages under this token limit"
        
        filter_passages = [p for p in passages if replace_with in p]
        assert all([self.example_string not in p for p in filter_passages]), "Bad manipulation; all patterns were not replaced"

class TestVectorstoreHandler:

    test_filename = "test_data.csv"
    test_vs_path = SAVE_PATH+"test/"

    example_string = "New York State Route"

    @pytest.fixture
    def vs(self):
        # make small for quick testing

        communicator = GPTCommunicator(api_key=API_KEY, model_name=MODEL_NAME)
        processor = WikiTextProcessor(
            dataset_version = "wikitext-2-raw-v1", 
            split = "train", 
            communicator = communicator,
            verbose = False
        )
        _ = processor.process_text(
            token_limit = 500, 
            save_path = SAVE_PATH,
            save_filename = self.test_filename,
        )
        vs = LangchainVectorstore(
            embedding_type = HuggingFaceEmbeddings(),
            processed_csv_path = SAVE_PATH+self.test_filename,
            verbose = False
        )
        return vs
    
    @pytest.fixture
    def communicator(self):
        return GPTCommunicator(api_key=API_KEY, model_name=MODEL_NAME)
    
    def test_create_and_retrieve(self, vs):

        n_retrieve = 2
        vs.create_local_vectorstore(save_path=self.test_vs_path, force_create=True)
        vs.create_retriever(search_kwargs={"k": n_retrieve})
        retrieved = vs.retrieve_top_documents(self.example_string)

        assert len(retrieved) == n_retrieve, "Failed to retrieve correct number of documents"

        # some strings may not always be in retrieved docs, but should be for this test case
        assert all([self.example_string in ret for ret in retrieved]), "Bad retrieved docuemnts"

    def test_load_and_retrieve(self, vs):

        n_retrieve = 2
        vs.load_local_vectorstore(load_path=self.test_vs_path)
        vs.create_retriever(search_kwargs={"k": n_retrieve})
        retrieved = vs.retrieve_top_documents(self.example_string)

        assert len(retrieved) == n_retrieve, "Failed to retrieve correct number of documents"

        # some strings may not always be in retrieved docs, but should be for this test case
        assert all([self.example_string in ret for ret in retrieved]), "Bad retrieved docuemnts"

    def test_text_chunking(self, vs, communicator):
        chunk_size = 250

        vs.chunk_data(chunk_size=chunk_size, chunk_overlap=0)
        assert max([communicator.count_tokens(vs.data[i].page_content) for i in range(len(vs.data))]) < chunk_size, "Bad text chunking"

class TestUtils:

    def test_update_user_config(self):

        # Fetch initial config file
        init_config = config.user_config
        init_config_val = init_config["TOKEN_LIMIT"]

        # Define a new random config val to update to
        new_config_val = random.randint(0, 10000)
        # if by chance we get same rand num, get new random number
        while new_config_val == init_config_val:
            new_config_val = random.randint(0, 10000)

        # Overwrite config.yml with new config
        new_config = {
            "TOKEN_LIMIT": new_config_val,
        }
        update_config_yml(new_config)

        # Fetch updated config file and compare with init config
        updated_config = config.user_config
        
        assert updated_config != init_config, "Failed to properly update config.yml"

    def test_update_patterns_config(self):

        # Clear file
        update_patterns_json(clear_json=True)

        # Fetch current patterns saved
        with open(PATTERNS_FILENAME) as f:
            init_patterns = json.load(f)

        # File should have been cleared
        assert init_patterns == {}, "Failed to properly clear patterns.json"
            
        # Create random patterns to update to
        random_key = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(25))
        random_val = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(25))

        # Overwrite patterns json
        update_patterns_json(random_key, random_val)

        # Fetch updated patterns file and compare with init
        with open(PATTERNS_FILENAME) as f:
            updated_patterns = json.load(f)

        assert init_patterns != updated_patterns, "Failed to properly update patterns.json"
        
class TestFactories:


    @pytest.fixture
    def communicator(self):
        return GPTCommunicator(api_key=API_KEY, model_name="gpt-3.5-turbo")
    
    def test_factory_create_wikitext_processor(self, communicator):

        # Init processor from factory
        dpf = DataProcessorFactory()
        processor = dpf.create_processor(name="WikiText", communicator=communicator)

        # Check returned object's class matches our module
        assert processor.__class__ == WikiTextProcessor, "Failed to properly create WikiText processor from factory"

    
    def test_factory_attach_vectorstore(self, communicator):

        # Make sure we can't post rag prompt without vectorstore
        with pytest.raises(ValueError):
            communicator.post_rag_prompt("Hi")

        # Attach vectorstore using factory
        vsf = VectorstoreFactory()
        vsf.attach_vectorstore("Langchain", communicator=communicator, load_vectorstore=True)

        # Make sure can perform rag now
        response, context = communicator.post_rag_prompt("Hi")
        assert isinstance(response, str) and isinstance(context, list), "Failed to get valid RAG response"

    def test_factory_create_model(self):

        # Test create non-rag model from factory
        model_factory = ModelFactory()
        model = model_factory.create_model("GPT_3.5_TURBO")

        # Make sure we can't post rag
        with pytest.raises(ValueError):
            response = model.post_rag_prompt("Hi")

        response = model.post_prompt("Hi")
        assert isinstance(response, str), "Failed to create proper model"

    def test_factory_create_rag_model(self):

        # Test create non-rag model from factory
        model_factory = ModelFactory()
        model = model_factory.create_model("GPT_3.5_TURBO_RAG")

        # Test we get valid RAG response
        response, context = model.post_rag_prompt("Hi")
        assert isinstance(response, str) and isinstance(context, list), "Failed to get valid RAG response"


    def test_bad_name(self, communicator):

        dpf = DataProcessorFactory()
        with pytest.raises(NotImplementedError):
            _ = dpf.create_processor(name="bad_name", communicator=communicator)

        vsf = VectorstoreFactory()
        with pytest.raises(NotImplementedError):
            vsf.attach_vectorstore("bad_name", communicator=communicator, load_vectorstore=True)

        model_factory = ModelFactory()
        with pytest.raises(NotImplementedError):
            _ = model_factory.create_model("bad_name")
