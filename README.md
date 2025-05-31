# RAG Based application to process queries related to given PDF document

### Retrieval Augmented Generation (RAG) System

---

## 🚀 Overview

This project implements a sophisticated Retrieval Augmented Generation (RAG) system designed to answer user queries based on information extracted from a PDF document. Built with **FastAPI** for a robust API interface and **LangChain** for orchestrating the RAG pipeline, it leverages **FAISS** for efficient similarity search and **Google Gemini models** for embeddings and response generation.

The system is particularly suited for knowledge base queries, like answering questions from a company handbook or FAQ documents, ensuring responses are grounded in provided context.

The system is also capable of handling edge cases where no query is passed and if the user query is not relevant to the PDf document.

PDF used for this project - (https://www.medicare.gov/Pubs/pdf/10050-medicare-and-you.pdf)

---

## ✨ Key Features

* **PDF Document Ingestion:** Processes PDF files, chunks them, and stores their embeddings in a FAISS vector database.
* **Intelligent Text Chunking:** Uses `RecursiveCharacterTextSplitter` with configurable `chunk_size` and `chunk_overlap` to maintain semantic coherence.
* **Dynamic K-Value Retrieval:** Adjusts the number of top-`k` relevant document chunks retrieved based on the complexity and intent of the user's query, optimizing context for the LLM.\
* **Confidence Scoring:** Provides a `confidence_score` with each LLM response, derived from the similarity of retrieved documents, indicating the system's certainty in its answer.
* **Structured JSON Responses:** Delivers LLM answers, confidence scores, and source document metadata (page, file name) in a clean JSON format.

---

## 🛠️ Technologies Used

* **Python**
* **FastAPI:** Web framework for building the API.
* **LangChain:** Framework for developing LLM-powered applications.
* **FAISS:** (Facebook AI Similarity Search) For efficient vector similarity search.
* **Google Gemini API:** For state-of-the-art text embeddings (`embedding-001`) and LLM (`gemini-2.0-flash`).
* **Pydantic:** Data validation and settings management.
* **python-dotenv:** For managing environment variables.

---

## 🚀 Setup and Installation

Follow these steps to get your project up and running:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Vishwaraj13/RAG_app.git
    ```

    OR

    ```
    Extract the .zip file provided
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv your_venv
    ```

3.  **Activate the Virtual Environment:**
    * **On Windows:**
        ```bash
        .\your_venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source your_venv/bin/activate
        ```

4.  **Install Dependencies:**
    Install all required Python packages.
    ```bash
    pip install -r requirements.txt
    ```
5.  **Configure Environment Variables:**
    Create a `.env` file in the root directory of your project and add your Google API Key:
    ```
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```
    You can obtain a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
---

## 🏃 Usage

Just run the following command to start the FastAPI endpoint

```uvicorn main:app --reload```

Upon running the application, PDF given will be loaded and split into chunks, converted into embeddings and stored in a newly created FAISS database
Afer the completion, you will get a message saying "Created FAISS vector database".

**Access the API Documentation:**

Swagger UI: http://127.0.0.1:8000/docs

**Query the RAG system:**

Example Request:
```
{
  "query": "how can i make a payment for medicare"
}
```
Example Response
```
{
  "answer": "There are four ways to pay your Medicare premium bill:\n\n1.  **Online through your secure Medicare account:** Visit Medicare.gov/account/login to log in (or create) your Medicare account. Then, select \\\"Pay my premium\\\" to make a payment by credit card, debit card, Health Savings Account (HSA) card, or from your checking or savings account. You’ll get a confirmation number when you make your payment. This service is free and is the fastest way to pay your premium.\n2.  **Through Medicare Easy Pay:** This free service automatically deducts your payment from your savings or checking account each month. Visit Medicare.gov/medicare-easy-pay, or call 1-800-MEDICARE (1-800-633-4227) to find out how to sign up. TTY users can call 1-877-486-2048.\n3.  **Through your bank:** Contact your bank to set up a one-time or recurring payment from your checking or savings account. Not all banks offer this service, and some charge a fee. Enter your information carefully to make sure your payment goes through on time. Give the bank this information:\n    *   Your 11-character Medicare Number: Enter the numbers and letters without dashes, spaces, or extra characters.\n    *   Payee name: CMS Medicare Insurance\n    *   Payee address: Medicare Premium Collection Center, PO Box 790355, St. Louis, MO 63179-0355\n    *   The amount of your payment\n4.  **Through the mail:** You can pay by check, money order, credit card, debit card, or HSA card. Fill out the payment coupon at the bottom of your bill and include it with your payment. Payments made by mail take longer to process than payments made quickly and securely through your online Medicare account. Use the return envelope that came with your bill, and mail your Medicare payment coupon and payment to: Medicare Premium Collection Center, PO Box 790355, St. Louis, MO 63179-0355",
  "source_page": "22,24,125",
  "confidence_score": 1,
  "chunk_size": 5386
}
```

## 📂 Project Structure
```
.
├── venv/                       # Python virtual environment
├── your_document.pdf          # PDF document
├── faq_rag_faiss_index/        # Directory containing the FAISS vector store files
│   └── index.faiss
│   └── index.pkl
├── .env                        # Environment variables (e.g., GOOGLE_API_KEY)
├── main.py                     # Main FastAPI application and RAG chain logic
├── requirements.txt            # Python dependencies
└── README.md                   # This file

```
