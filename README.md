# CORD-19  Extraction Agent

## Overview

This project implements an AI agent that processes the COVID-19 Open Research Dataset (CORD-19) to generate document summaries and extract keywords. The system uses deep learning with TensorFlow and stores document embeddings in a vector database for efficient similarity search.

## Features

- **Data Processing**: Downloads and processes the CORD-19 metadata dataset from Kaggle
- **Document Embeddings**: Creates vector embeddings using TensorFlow Universal Sentence Encoder
- **Vector Storage**: Stores document embeddings in a FAISS vector database for fast similarity search
- **Summarization**: Generates concise summaries of research papers using the T5 transformer model
- **Keyword Extraction**: Identifies key topics and keywords using NMF on TF-IDF vectors
- **Interactive UI**: Provides a Streamlit web application for exploration and demonstration

## Architecture

The system consists of several components:

1. **Data Loader**: Downloads and processes the CORD-19 dataset from Kaggle
2. **Embedding Engine**: Creates vector embeddings for documents using TensorFlow
3. **Vector Store**: Stores embeddings in FAISS for efficient similarity search
4. **Summarization Module**: Generates summaries using the T5 transformer model
5. **Keyword Extractor**: Identifies keywords and topics using NMF
6. **Search Engine**: Enables semantic search based on vector similarity
7. **Interactive UI**: Provides a user-friendly interface for exploration

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Kaggle API credentials (for dataset download)
- Sufficient disk space (~5GB) for the dataset and models

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/cord19-summarizer.git
   cd cord19-summarizer
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up Kaggle credentials (if not already done):
   ```
   kaggle config set -n username -v your_username
   kaggle config set -n key -v your_key
   ```

### Usage

#### Core Agent API

To use the core agent in your Python code:

```python
from cord19_summarizer import CORD19Agent

# Initialize the agent
agent = CORD19Agent()

# Download the dataset
agent.download_dataset()

# Load metadata
metadata = agent.load_metadata()

# Create document embeddings and vector store
documents, embeddings = agent.create_document_embeddings()

# Extract keywords
topics = agent.extract_keywords(documents)

# Generate summaries
summaries = agent.generate_batch_summaries(documents, sample_size=5)

# Search for similar papers
similar_docs = agent.search_similar_documents("covid-19 transmission in indoor environments")
```

#### Interactive Application

To launch the interactive Streamlit application:

```
streamlit run cord19_app.py
```

This will start a local web server and open the application in your default browser.

## Technical Details

### Embedding Model

The system uses the Universal Sentence Encoder from TensorFlow Hub to generate document embeddings. This model encodes text into high-dimensional vectors that capture semantic meaning.

### Summarization Model

Document summarization is performed using the T5 (Text-to-Text Transfer Transformer) model, which is trained to generate concise summaries of longer texts.

### Keyword Extraction

Keywords are extracted using Non-negative Matrix Factorization (NMF) on TF-IDF vectors, which identifies important topics and their associated terms.

### Vector Storage

Document embeddings are stored in a FAISS vector database, which provides efficient similarity search capabilities for large-scale document collections.

## Dataset

The COVID-19 Open Research Dataset (CORD-19) is a resource of scientific papers about COVID-19 and related coronaviruses, created to facilitate research. The dataset includes metadata about papers such as titles, authors, abstracts, and publication dates.

## Future Improvements

- Implement more advanced summarization techniques like recursive chunking for long documents
- Add support for multi-language processing
- Incorporate citation graph analysis for research impact assessment
- Implement document clustering for better topic organization
- Add sentiment analysis for research trend evaluation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Allen Institute for AI for providing the CORD-19 dataset
- Google for the Universal Sentence Encoder and TensorFlow
- Hugging Face for the transformers library
