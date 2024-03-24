from constants import MODEL, RETRIEVE_CONTEXT_FOR_ANSWERS, SYSTEM_PROMPT, TEXTBOOK_PATH
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import logging
import os


class HIPAgent:
  """
  Agent to perform Multiple Choice Question Answering (MCQA) by querying GPT 3.5.

  Performance is addtionally improved using Retrieval Augmented Generation (RAG) where
  additional context from a relevant document, in this case Biology 2e, is passed to
  the model.

  The RAG code here is heavily inspired by the examples in the Langchain cookbook     
  (https://python.langchain.com/docs/expression_language/cookbook/).
  """

  def __init__(self, use_rag: bool = True) -> None:
    self.openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    self.vector_db = None
    self.vector_retriever = None
    if use_rag:
      self._initialize_retriever()

    # Configure logging handlers.
    self.logger = logging.getLogger('hip_agent')
    handler = logging.FileHandler('app.log')
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s — %(levelname)s — %(message)s"))
    self.logger.addHandler(handler)
    self.logger.setLevel(logging.DEBUG)

  def _initialize_retriever(self) -> None:
    """
    Initialize the retriever for RAG.

    We use Facebook AI Similarity Search (FAISS) as a vector store for the embeddings.
    """
    # Load and create embeddings of the biology textbook for RAG.
    loader = TextLoader(TEXTBOOK_PATH)
    biology_textbook = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=150)
    docs = text_splitter.split_documents(biology_textbook)
    embeddings_model = OpenAIEmbeddings(
        openai_api_key=os.environ['OPENAI_API_KEY'])
    # Initialize the vector store.
    self.vector_db = FAISS.from_documents(docs, embeddings_model)
    self.vector_retriever = self.vector_db.as_retriever()

  def _retrieve_context(self, question: str, answer_choices: list[str]) -> str:
    """
    Retrieve passages from the biology textbook that are semantically similar to the
    question to pass as additional context to the model.
    """
    match_docs = self.vector_retriever.get_relevant_documents(question)
    if RETRIEVE_CONTEXT_FOR_ANSWERS:
      # Retrieve embeddings for the various answer chocies as well and pass this as
      # additional context to the model. This is gated by a flag because, sometimes,
      # this leads to slightly poorer performance on this testbench.
      for answer_choice in answer_choices:
        match_docs.extend(
            self.vector_retriever.get_relevant_documents(answer_choice))
    matches = '\n'.join([m.page_content for m in match_docs])
    return matches

  def _format_options(self, answer_choices: list[str]) -> str:
    """
    Format the possible answers into a format aligned with the Sytem Prompt.

    Eg.
    Option A: ...
    Option B: ... 
    """
    # We use chr() to convert an index of "0" to "Option A"
    options = [
      f'Option {chr(65+index)}: {choice}'
      for index, choice in enumerate(answer_choices)
    ]
    return "\n".join(options)
    
  def _get_index_of_response(self, response: str, answer_choices: list[str]) -> int:
    """
    Match the answer from the model to an index in answer_choices.
    
    The response from the model is usually of the format "Answer: D". 
    
    Sometimes the model can be a little verbose and try to explain its response.
    Eg. Answer A. When individuals mate with those who are similar to themselves.
    """
    response = response.lstrip("Answer: ")
    response = response.lstrip("Option ")
    
    # Try to find the index of the response in the answer choices using
    # the option's letter index.
    option = response.split(':')[0]
    if len(option) == 1 and ord(option)-ord('A') < len(answer_choices):
      return ord(option) - ord('A')
      
    # If the above strategy doesn't work, try to find a match between the answer choices
    # and the text of the response.
    for index, answer_choice in enumerate(answer_choices):
      if answer_choice in response:
        return index

    # If we haven't found any match, return -1.
    self.logger.error(
      "Cannot find match for {response_text} among {answer_choices}")
    return -1

  def get_response(self, question: str, answer_choices: list[str]) -> int:
    """
    Calls the OpenAI 3.5 API to generate a response to a multiple choice question.
    The response is then matched to one of the answer choices and the index of the
    matching answer choice is returned. If the response does not match any answer choice,
    -1 is returned.

    Args:
        question: The question to be asked.
        answer_choices: A list of answer choices.

    Returns:
        The index of the answer choice that matches the response, or -1 if the response
        does not match any answer choice.
    """
    # Retrieve the background context from the textbook for RAG.
    context = self._retrieve_context(question, answer_choices)

    # Format the possible answers into the format aligned with the Sytem Prompt.
    answer_choices_str = self._format_options(answer_choices)

    # Create the prompt.
    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }, {
        "role":
        "user",
        "content":
        f"Background Context:\n{context}\nQuestion: {question}.\nPossible Answers: {answer_choices_str}\nAnswer: "
    }]

    # Call the OpenAI 3.5 API.
    response = self.openai_client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    response_text = response.choices[0].message.content
    self.logger.debug(f'Question: {question} Model Answer: {response_text}')
    return self._get_index_of_response(response_text, answer_choices)

