# Custom RAG System

A lightweight, custom-built Retrieval-Augmented Generation (RAG) system built with Python. This system provides document-based question answering without external dependencies like LangChain or LangGraph.

## 🚀 Features

- **Multi-format Document Support**: TXT, PDF, DOCX files
- **Smart Text Chunking**: Intelligent document splitting with overlap
- **OpenAI Embeddings**: Uses OpenAI's text-embedding-ada-002 model
- **Vector Similarity Search**: Semantic search through document chunks
- **Fallback Text Search**: Basic keyword matching when embeddings fail
- **Interactive Interface**: Command-line Q&A system
- **Source Attribution**: Always shows where answers come from

## 🏗️ Architecture

```
Document Loader → Text Chunker → Embedding System → Vector Store → Query Interface
      ↓              ↓              ↓              ↓            ↓
   Load files    Split chunks   Generate      Store &      Search &
   (TXT/PDF/    with overlap   embeddings    search       answer
    DOCX)
```

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for API calls

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd rag-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Add documents:**
   Place your documents in the `data/` folder

## 🚀 Usage

### Basic Usage

1. **Start the system:**
   ```bash
   python run.py
   ```

2. **Ask questions interactively:**
   ```
   Enter your question: What is this document about?
   ```

3. **Exit the system:**
   ```
   Enter your question: quit
   ```

### Supported File Formats

- **Text Files (.txt)**: Plain text documents
- **PDF Files (.pdf)**: Multi-page PDF documents
- **Word Documents (.docx)**: Microsoft Word files

### Example Workflow

1. **Upload documents** to `data/` folder
2. **Run the system** with `python run.py`
3. **Ask questions** about your documents
4. **Get answers** with source citations

## 📁 Project Structure

```
rag-system/
├── src/
│   ├── main.py              # Main application entry point
│   ├── document_loader.py   # Document loading and parsing
│   ├── text_chunker.py      # Text splitting and chunking
│   ├── embedding_system.py  # OpenAI embeddings generation
│   ├── vector_store.py      # Vector storage and similarity search
│   └── rag_pipeline.py      # Main RAG orchestration
├── data/                    # Document storage folder
├── requirements.txt         # Python dependencies
├── run.py                  # Quick start script
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Customization

- **Chunk Size**: Modify `chunk_size` in `text_chunker.py`
- **Overlap**: Adjust `overlap` in `text_chunker.py`
- **Search Results**: Change `top_k` parameter in query methods
- **Models**: Update OpenAI model names in the code

## 🎯 How It Works

1. **Document Loading**: System scans `data/` folder for supported files
2. **Text Extraction**: Converts documents to plain text
3. **Chunking**: Splits text into manageable chunks with overlap
4. **Embedding**: Generates vector representations using OpenAI
5. **Storage**: Stores chunks and embeddings in memory
6. **Querying**: Searches for relevant chunks using similarity
7. **Response**: Generates answers based on retrieved context

## 🚨 Limitations

- **Memory-based storage**: Data is lost when the program stops
- **API dependency**: Requires OpenAI API access
- **Document scope**: Only knows what's in your uploaded documents
- **No persistent storage**: Vector database is not saved to disk

## 🔮 Future Enhancements

- [ ] Persistent vector database storage
- [ ] Web interface
- [ ] Batch processing for large document collections
- [ ] Multiple embedding model support
- [ ] Document update and versioning
- [ ] Export/import functionality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- OpenAI for providing the embedding and completion APIs
- The open-source community for inspiration and tools

## 📞 Support

If you encounter any issues or have questions:

1. Check the existing issues
2. Create a new issue with detailed information
3. Include error messages and system information

---

**Happy Document Searching! 🎉**
