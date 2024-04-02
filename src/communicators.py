""" Classes for LLM API communication. """

import logging
from abc import ABC, abstractmethod
from typing import Optional

import tiktoken
from openai import OpenAI

from src.vectorstore_handlers import VectorstoreHandler

# Define a system RAG role
RAG_SYS_ROLE_MSG = "You will answer user queries based on the context documents provided. Your responses MUST be grounded from the provided context.\
YOU WILL LIMIT YOUR KNOWLEDGE ONLY TO THE INFORMATION PROVIDED. YOU WILL NOT PROVIDE ANY EXTERNAL INFORMATION. \
If information needed to answer the user query is not in the documents provided, you will reply with 'Sorry, I can't answer that based on the provided documents'."

# Define messages to preface context and query when constructing prompt
RAG_CONTEXT_PREFACE = "Please use the following context to generate your response, which must not contain outside information:\n\n"
RAG_QUERY_PREFACE = "\n\nBased solely on the context provided above, please answer the following user query:\n"


# Decorator for methods requiring vectorstore to be set
def vs_required(function):
    def wrapper(self, *args, **kwargs):
        if self.vs_hndlr is None:
            raise ValueError(
                "vs_hndlr not set; pass VectostoreHandler upon init or invoke set_vectorstore_handler() before using this method."
            )
        return function(self, *args, **kwargs)

    return wrapper


class Communicator(ABC):
    """Base class for Communicator subclasses"""

    def __init__(self, vectorstore_handler: Optional[VectorstoreHandler] = None):
        """Init with optional Vectorstore Handler."""
        self.vs_hndlr = vectorstore_handler

    def set_vectorstore_handler(
        self, vectorstore_handler: Optional[VectorstoreHandler] = None
    ):
        """A method to set the vectorstore handler post-init."""
        self.vs_hndlr = vectorstore_handler

    @abstractmethod
    def post_prompt(self):
        """A method to send/receive messages with LLM, whether through API post or local process."""
        pass

    @abstractmethod
    def count_tokens(self):
        """A method to count tokens in text using respective LLM's tokenizer."""
        pass

    @abstractmethod
    def post_rag_prompt(self):
        """A method for posting with RAG; add vs_required decorator when implementing in subclass"""
        pass


class GPTCommunicator(Communicator):
    """Communicator subclass for communication with OpenAI GPT models."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        vectorstore_handler: Optional[VectorstoreHandler] = None,
    ) -> None:
        """Init object

        Args:
            api_key (str): OpenAI API access token
            model_name (str, optional): GPT version to use; defaults to "gpt-3.5-turbo".
            vectorstore_handler (Optional[VectorstoreHandler]): Vectorstore handler object; needed if invoking RAG method

        Raises:
            ValueError: Raised when model_name is not a valid GPT model.
        """
        super().__init__(vectorstore_handler)
        # init client with api key
        self.client = OpenAI(api_key=api_key)

        # context window limits; found at https://platform.openai.com/docs/models
        model_max_tokens = {
            "gpt-3.5-turbo": 16385,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
        }

        # check for valid model name input
        if model_name not in model_max_tokens.keys():
            raise ValueError(
                f"Invalid model name; valid args include: {model_max_tokens.keys()}"
            )
        self.model_name = model_name

        # set model attributes
        self.max_prompt_tokens = (
            model_max_tokens[model_name] - 250
        )  # buffer for response tokens
        self.system_role = "You are a helpful AI assistant."  # default role
        self.total_tokens_used = 0
        self.temperature = (
            0  # keep as 0 to minimize responses straying from provided documents
        )

    def post_prompt(self, text: str, truncate: bool = True) -> Optional[str]:
        """Method to communicate with GPT.

        Args:
            text (str): Input text prompt to send to GPT.
            truncate (bool): If true, will truncate the text to model's token limit before sending.

        Returns:
            Optional[str]: GPT's text response; None if fails.
        """
        try:
            if truncate:
                text = self.truncate_text(text, token_limit=self.max_prompt_tokens)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": str(self.system_role)},
                    {"role": "user", "content": str(text)},
                ],
                temperature=self.temperature,
            )
            self.last_response = response
            self.total_tokens_used += int(response.usage.total_tokens)

        except Exception as e:
            logging.error(f"Failed to post prompt: {e}")
            return None

        return response.choices[0].message.content

    def count_tokens(self, text: str) -> Optional[int]:
        """Method to count number of tokens in a given piece of text.

        Args:
            text (str): Input text to count tokens from.

        Returns:
            Optional[int]: Token count; None if fails.
        """
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
            num_tokens = len(encoding.encode(text))
        except Exception as e:
            logging.error(f"Failed to count tokens: {e}")
            return None

        return num_tokens

    def truncate_text(self, text: str, token_limit: int) -> Optional[str]:
        """Method to truncate a string to a specified token limit.
        Parses by sentence period to not truncate in the middle of a sentence.

        Args:
            text (str): Input text to truncate.
            token_limit (int): Number of tokens to truncate text to.

        Returns:
            Optional[str]: Truncated text; None if try fails.
        """
        token_count = 0
        truncated_text = ""
        try:
            for line in text.split("."):
                if line.strip() in [""]:
                    continue

                line = line + "."  # add . back after split

                token_count += self.count_tokens(line)
                if token_count >= token_limit:
                    break

                truncated_text += line

            truncated_text += "\n"

        except Exception as e:
            logging.error(f"Failed to truncate text: {e}")
            return None

        return truncated_text

    @vs_required
    def post_rag_prompt(self, query: str):
        """Method to communicate with GPT using RAG-constructed prompts.

        Args:
            query (str): Input question ask GPT.

        Returns:
            Optional[str]: GPT's text response; None if try fails.
        """

        if self.system_role != RAG_SYS_ROLE_MSG:
            self.system_role = RAG_SYS_ROLE_MSG

        # Retrieve top matched documents from vectorstore
        top_context = self.vs_hndlr.retrieve_top_documents(query)

        # Combine list of context docuements into a single string and add preface messages
        all_context = RAG_CONTEXT_PREFACE + "\n\n".join(top_context)
        query = RAG_QUERY_PREFACE + query

        # Cut context down if needed; account for size of query
        buffer_token_space = self.count_tokens(query)
        token_limit = self.max_prompt_tokens - buffer_token_space

        if self.count_tokens(all_context) > token_limit:
            prompt = self.truncate_text(all_context, token_limit) + query
        else:
            prompt = all_context + query

        response = self.post_prompt(
            prompt
        )  # can't use truncate arg here, since it would cut our query at the bottom

        return response, top_context
