""" Classes for LLM communication. """

from openai import OpenAI
import tiktoken
import logging
from typing import Optional
from abc import ABC, abstractmethod

class Communicator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def post_prompt(self):
        pass

    @abstractmethod
    def count_tokens(self):
        pass

    @abstractmethod
    def truncate_text(self):
        pass


class GPTCommunicator(Communicator):

    def __init__(
            self, api_key_path: str, model_name: str = "gpt-3.5-turbo",
        ) -> None:
        """ Init object 

        Args:
            api_key_path (str): File path to OpenAI API key, stored as a txt file.
            model_name (str, optional): GPT version to use; defaults to "gpt-3.5-turbo".

        Raises:
            ValueError: Raised when model_name is not a valid GPT model.
        """        
        super().__init__()
        # init client with api key file
        with open(api_key_path) as f:
            self.client = OpenAI(api_key=f.readline().strip())
        
        # context window limits; found at https://platform.openai.com/docs/models
        model_max_tokens = { 
            "gpt-3.5-turbo": 16385,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
        }

        # check for valid model name input
        if model_name not in model_max_tokens.keys():
            raise ValueError(f"Invalid model name; valid args include: {model_max_tokens.keys()}")
        self.model_name = model_name

        # set model attributes
        self.max_prompt_tokens = model_max_tokens[model_name] -  250 # buffer for response tokens
        self.system_role = "You are a helpful AI assistant." # default role
        self.total_tokens_used = 0
        
    def post_prompt(self, text: str, truncate: bool=True) -> Optional[str]:
        """ Method to communicate with GPT.

        Args:
            text (str): Input text prompt to send to GPT.
            truncate (bool): If true, will truncate the text to model's token limit before sending.

        Returns:
            Optional[str]: GPT's text response; None if try fails. 
        """
        try:
            if truncate:
                text = self.truncate_text(text, token_limit=self.max_prompt_tokens)

            response = self.client.chat.completions.create(
                model = self.model_name,
                messages = [
                    {"role": "system", "content": str(self.system_role)},
                    {"role": "user", "content": str(text)}
                ]
            )
            self.last_response = response
            self.total_tokens_used += int(response.usage.total_tokens)

        except Exception as e:
            logging.error(f"Failed to post prompt: {e}")
            return None
        
        return response.choices[0].message.content
    
    def count_tokens(self, text: str) -> Optional[int]:
        """ Method to count number of tokens in a given piece of text. 

        Args:
            text (str): Input text to count tokens from.

        Returns:
            Optional[int]: Token count; None if try fails.
        """        
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
            num_tokens = len(encoding.encode(text))
        except Exception as e:
            logging.error(f"Failed to count tokens: {e}")
            return None

        return num_tokens
    
    def truncate_text(self, text: str, token_limit: int) -> Optional[str]:
        """ Method to truncate a string to a specified token limit.

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

                line = line + "." # add . back after split

                token_count += self.count_tokens(line)
                if token_count >= token_limit:
                    break

                truncated_text += line

            truncated_text += "\n"

        except Exception as e:
            logging.error(f"Failed to truncate text: {e}")
            return None
        
        return truncated_text