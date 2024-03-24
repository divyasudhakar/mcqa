# Multiple Choice Question Answering using GPT 3.5
This library runs a multiple choice question answering test set against GPT 3.5. The code uses Retrieval Augmented Generation (RAG) where additional context from a relevant document, in this case Biology 2e, is passed to the model in order to improve performance and reduce hallucinations.

## Setting up the service
The service requires python major version 3 and above to run as well as a few libraries.

You can run `make setup` to install all the necessary libraries.

## Running the service

Run `make run`.

This should query the model and output a score at the end. 

At last run, the score was 18/20 on the provided test bench.

## Running unit tests

Run `make test`
