# Jal-Sandhana-SIH24-Project
Jal Sandhana is an AI-based chatbot designed to provide concise and informative answers related to groundwater. It assists users in understanding groundwater levels, hydrogeological scenarios, water quality, groundwater resource assessment, and NOC (No Objection Certificate) requirements for groundwater extraction in India.


# Groundwater Chatbot - Jal Sandhana

## Overview
Jal Sandhana is an AI-based chatbot designed to provide concise and informative answers related to groundwater. It assists users in understanding groundwater levels, hydrogeological scenarios, water quality, groundwater resource assessment, and NOC (No Objection Certificate) requirements for groundwater extraction in India.

## Features
- Extracts relevant data from PDFs and stores it in a FAISS-based vector database.
- Uses Hugging Face embeddings to improve search and retrieval accuracy.
- Integrates with Groq's API to generate AI-driven responses.
- Implements a FastAPI-based backend with CORS support for secure API access.

## Tech Stack
- **Backend Framework**: FastAPI
- **Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Hugging Face's `sentence-transformers/all-MiniLM-L6-v2`
- **LLM API**: Groq with LLaMA 3 Model
- **Document Processing**: PyPDFLoader
- **Environment Management**: dotenv

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/pushpakgoel621/Jal-Sandhana-SIH24-Project.git
   cd <repository-folder>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add the following variables:
     ```ini
     GROQ_API_KEY=<your-groq-api-key>
     ```
4. Run the application:
   ```sh
   uvicorn main:app --reload
   ```

## Usage
### API Endpoints
- **`GET /docs`** - Access API documentation via Swagger UI.
- **`POST /query`** - Send a query related to groundwater and receive a response.

### Adding PDF Documents
Place the relevant PDF files in the `data/` folder. The system will automatically process and store them in the vector database.

## Project Structure
```
.
├── data/                    # Folder for PDF documents
├── vectorstore/             # Directory for FAISS database
├── main.py                  # FastAPI backend script
├── .env                     # Environment variables
├── requirements.txt         # Required dependencies
└── README.md                # Project documentation
```

## Contributing
1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit changes and push to your fork.
4. Open a pull request.

## License
This project is licensed under the MIT License.

