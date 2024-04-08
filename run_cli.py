"""Script to create and interact with RAG model through command line interface."""

import logging
import sys
import os

# Avoid huggingface tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def init_rag_model():
    """Create the RAG model using user_config.yml values."""

    # Import here, since user may adjust values after script starts
    from src.config import config
    from src.factories import ModelFactory
    from src.app_helpers import get_model_factory_name

    model_name = config.user_config["MODEL_NAME"]
    dataset_name = config.user_config["DATASET_NAME"]
    vectorstore_name = config.user_config["VECTORSTORE_NAME"]

    mf = ModelFactory()
    model = mf.create_model(
            get_model_factory_name(model_name, rag=True), 
            dataset_name=dataset_name,
            vectorstore_name=vectorstore_name,
            new_vectorstore=True,
        )
    
    return model


def main():
    """Conduct main script logic."""

    # Init logging with info messages during model creation
    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO,
    )

    # Pause for user to set config params
    input("Set run params in src/config/user_config.yml and src/config/manipulate_patterns.json before proceeding. Press Enter to continue.")

    # Initialize model with defined config
    model = init_rag_model()

    # Switch logging back to error messages only during conversation
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.ERROR,
    )

    print("\n\nModel initialized and ready for QA. Type 'q' or 'quit' to end the conversation.")

    # Init with placeholder value for while loop
    user_query = "query"

    while user_query.lower().strip() not in ['q', 'quit']:
        user_query = input("\nInput query: ")

        if user_query in ['q', 'quit']:
            sys.exit() # exit to avoid an unnecessary model call
        
        response, context = model.post_rag_prompt(user_query)
        print(f"\nModel response: {response}")

if __name__ == "__main__":

    main()

