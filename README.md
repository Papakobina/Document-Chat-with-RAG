# Document-Chat-with-RAG

## Overview

This project uses the power of AI to retrieve and process data. It leverages the OpenAI API for embeddings and chat, and Pinecone for vector storage. The main goal of this project is to create a system that can answer queries about documents provided to it using and implementing RAG. For learning purposes I first implemented this functionality using pinecone to store vectors in the cloud and later used "FAISS" to store vectors Locally

## Features

- **OpenAI Embeddings**: This project uses OpenAI's powerful language model to generate embeddings for text data.
- **ChatOpenAI**: We use OpenAI's chat models to generate responses to prompts.
- **Pinecone Vector Store**: We use Pinecone, a vector database, for storing and retrieving vector embeddings.
- **Data Retrieval and Processing**: The project includes a pipeline for retrieving and processing data.

## Main Script Explanation

The `main.py` script is the entry point of the project. It orchestrates the retrieval and processing of data, and the interaction with the chat model. Here's a detailed explanation of each step:

1. **Setting Up**: The script starts by initializing the `OpenAIEmbeddings` and `ChatOpenAI` objects. The `OpenAIEmbeddings` object is used to generate embeddings for text data, and the `ChatOpenAI` object is used to generate responses to prompts.

2. **Querying**: A query is defined, and a `PromptTemplate` is created from this query. The `PromptTemplate` is then passed to the `ChatOpenAI` object to generate a response.

3. **Vector Store Setup**: The `PineconeVectorStore` object is initialized with the name of the Pinecone index and the `OpenAIEmbeddings` object. This object is used to store and retrieve vector embeddings.

4. **Retrieval QA Chat Prompt**: The `retrieval_qa_chat_prompt` is pulled from the `langchain-ai/retrieval-qa-chat` repository. This prompt is used to generate responses to queries.

5. **Document Combination Chain**: The `create_stuff_documents_chain` function is called to create a chain of operations for combining documents. This chain includes the `ChatOpenAI` object and the `retrieval_qa_chat_prompt`.

6. **Retrieval Chain**: The `create_retrieval_chain` function is called to create a chain of operations for retrieving documents. This chain includes the `PineconeVectorStore` object as a retriever and the document combination chain.

7. **Retrieval Invocation**: The retrieval chain is invoked with the query as input. The result is a set of documents that are relevant to the query.

8. **Template Creation**: A template is created for formatting the context and question. This template is used to generate the final response to the query.

This script demonstrates how to use the various components of the project to retrieve and process data, and to generate responses to queries. It provides a good starting point for understanding how the project works.

## Ingestion Script

The `ingestion.py` script is responsible for loading, processing, and storing the data. Here's a brief overview of what it does:

1. **Load Environment Variables**: The script starts by loading environment variables from a `.env` file. These variables include the OpenAI API key and the name of the Pinecone index.
2. **Load Data**: The script uses the `TextLoader` class from the `langchain_community.document_loaders` module to load a text file. The path to the file and its encoding are specified when creating the `TextLoader`.
3. **Split Text**: The script uses the `CharacterTextSplitter` class from the `langchain_text_splitters` module to split the loaded document into chunks. The size of the chunks and the overlap between them can be specified when creating the `CharacterTextSplitter`.
4. **Generate Embeddings**: The script uses the `OpenAIEmbeddings` class from the `langchain_openai` module to generate embeddings for the text chunks. The OpenAI API key is passed to the `OpenAIEmbeddings` constructor.
5. **Store Embeddings**: Finally, the script uses the `PineconeVectorStore` class from the `langchain_pinecone` module to store the embeddings in a Pinecone index. The name of the index is passed to the `PineconeVectorStore.from_documents` method.

To run the script, use the command `python ingestion.py`.

## Technical Explanation

This project involves several steps to process and store data. Here's a detailed explanation of each step:

1. **Loading Environment Variables**: The script starts by loading environment variables from a `.env` file using the `load_dotenv` function from the `dotenv` module. These variables include the OpenAI API key and the name of the Pinecone index.

2. **Loading Data**: The script uses the `TextLoader` class from the `langchain_community.document_loaders` module to load a text file. The path to the file and its encoding are specified when creating the `TextLoader`. The `load` method is then called to read the file and return its contents as a string.

3. **Splitting Text**: The script uses the `CharacterTextSplitter` class from the `langchain_text_splitters` module to split the loaded document into chunks. The `chunk_size` parameter determines the number of characters in each chunk, and the `chunk_overlap` parameter determines the number of characters that consecutive chunks have in common. The `split_documents` method is called to perform the splitting.

4. **Generating Embeddings**: The script uses the `OpenAIEmbeddings` class from the `langchain_openai` module to generate embeddings for the text chunks. An embedding is a vector that represents the semantic content of a piece of text. The OpenAI API key is passed to the `OpenAIEmbeddings` constructor, and the embeddings are generated by calling the `embed` method for each text chunk.

5. **Storing Embeddings**: Finally, the script uses the `PineconeVectorStore` class from the `langchain_pinecone` module to store the embeddings in a Pinecone index. A vector index is a data structure that allows for efficient similarity search and retrieval of vectors. The `from_documents` method is called to store the embeddings, with the text chunks, embeddings, and index name passed as arguments.

This process allows us to transform raw text data into a form that can be efficiently searched and retrieved based on semantic content. This is useful for a wide range of applications, including information retrieval, recommendation systems, and natural language processing tasks.

## Setup

1. Clone the repository.
2. Install the required dependencies.
3. Set up your environment variables. You'll need to provide your OpenAI API key and the name of your Pinecone index.
4. Run `main.py` to start the program.

## Usage

To use the project, run `main.py`. The script will retrieve data, generate embeddings using OpenAI, and store the embeddings in Pinecone. It will then use a chat model to generate a response to a prompt.

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

## License

This project is licensed under the terms of the MIT license.
