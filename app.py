import gradio as gr
import pdfplumber
import chromadb
from google import genai
from google.genai import types
from google.api_core import retry
import os

# Load API key from Hugging Face secrets
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API Key is missing. Set it as a secret in Hugging Face.")

# Initialize Google Gemini Client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Define retriable API calls
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})


class GeminiEmbeddingFunction:
    def __init__(self, document_mode=True):
        self.document_mode = document_mode

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input):
        task_type = "retrieval_document" if self.document_mode else "retrieval_query"
        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in response.embeddings]


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    return text.strip()


# Init ChromaDB
DB_NAME = "document_db"
embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True
chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)


def upload_pdf(pdf_file):
    if pdf_file is None:
        return "Please upload a PDF.", gr.update(value=None)
    
    # Extract text
    pdf_text = extract_text_from_pdf(pdf_file)

    # Add to ChromaDB
    db.delete(ids=["doc1"])  # Clear previous document
    db.add(documents=[pdf_text], ids=["doc1"])

    return "PDF uploaded and embeddings stored!", True


def answer_question(user_question, embeddings_ready):
    if not embeddings_ready:
        return "Please upload a PDF first."

    if user_question.strip() == "":
        return "Please enter a question."

    embed_fn.document_mode = False
    result = db.query(query_texts=[user_question], n_results=1)
    retrieved_texts = result["documents"][0] if "documents" in result else []

    if not retrieved_texts:
        return "Sorry, I couldn't find relevant information in the document."
    # Generate answer using Google Gemini
    prompt = f"""
    You are a helpful assistant that answers questions using the reference text provided.
    Answer comprehensively but simply. If the passage is irrelevant, state that.
    
    QUESTION: {user_question}
    """
    for passage in retrieved_texts:
        prompt += f"PASSAGE: {passage}\n"

    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    
    return response.text if response else "Sorry, I couldn't generate an answer."

    # return retrieved_texts[0]


# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("## 📄 AI PDF Q&A using Google Gemini")
    gr.Markdown("Step 1: Upload a PDF document\nStep 2: Ask questions about its content!")

    embeddings_ready = gr.State(False)

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", type="filepath")
        upload_status = gr.Textbox(label="Upload Status", interactive=False)
    
    upload_button = gr.Button("Upload & Store Embeddings")
    upload_button.click(fn=upload_pdf, inputs=pdf_input, outputs=[upload_status, embeddings_ready])

    with gr.Row():
        question_input = gr.Textbox(label="Ask a Question", placeholder="e.g. What is the procedure to defrost?")
        answer_output = gr.Textbox(label="Answer", interactive=False)

    ask_button = gr.Button("Get Answer")
    ask_button.click(fn=answer_question, inputs=[question_input, embeddings_ready], outputs=answer_output)

app.launch()
