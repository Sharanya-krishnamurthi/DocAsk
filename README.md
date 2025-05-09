# 🤖 DocAsk – AI PDF Q\&A with Google Gemini

**DocAsk** is an intelligent web app that allows you to upload any PDF document and ask natural-language questions about its content. It uses Google Gemini for embeddings and question-answering, and ChromaDB for document retrieval.

---

## ✨ Features

* 📥 Upload a PDF and automatically extract its text.
* 🧠 Embed document contents using **Gemini embeddings**.
* 🔎 Ask questions and get contextual answers.
* 🧾 Real-time semantic search powered by **ChromaDB**.
* ⚡ Fast and accurate answers using **Gemini 2.0 Flash** model.

---

## 📦 Requirements

Before you begin, make sure to install dependencies:

```bash
pip install gradio pdfplumber chromadb google-generativeai
```

---

## 🔑 API Key Setup

This app requires a **Google Generative AI API key**. Set it as an environment variable:

```bash
export GOOGLE_API_KEY=your_api_key_here
```

Or in Python:

```python
os.environ["GOOGLE_API_KEY"] = "your_api_key_here"
```

---

## 🚀 Usage

To run the app:

```bash
python app.py
```

Once running, a browser window will open where you can:

1. Upload a PDF document.
2. Ask any question related to the document.
3. Receive contextual and human-like answers.

---

## 🔍 How It Works

1. **PDF Text Extraction**: Uses `pdfplumber` to extract text from uploaded files.
2. **Embedding**: Text is converted into embeddings using `text-embedding-004` from Gemini.
3. **Vector Storage**: Embeddings are stored and queried using `chromadb`.
4. **Answer Generation**: Google Gemini generates a response based on the retrieved passage and user query.

---

## 📁 File Structure

```text
├── app.py                # Main application
├── requirements.txt      # Required libraries
└── README.md             # You're here!
```

---

## ⚠️ Notes

* Max PDF size may vary based on the deployment platform.
* Gemini's models require valid API credentials and have rate limits.
* This project uses in-memory ChromaDB – not persistent between sessions unless configured otherwise.

---

## 🧑‍💻 Author

Built with ❤️ using Gemini, Gradio, and ChromaDB by \Sharanya Krishnamurthi.

---

## DEMO
[HuggingFace Spaces](https://huggingface.co/spaces/sharanya/DocQA)

