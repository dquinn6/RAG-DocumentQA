import pytest
from src.communicators import GPTCommunicator
from src.factories import DataProcessorFactory, VectorstoreFactory, ModelFactory
from src.data_processors import WikiTextProcessor
from src.config import config

API_KEY = config.user_config["ACCESS_TOKEN"]

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
