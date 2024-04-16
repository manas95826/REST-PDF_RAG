from flask import Flask, request, jsonify
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from pdfminer.high_level import extract_text
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Set Hugging Face API token
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

# Function to preprocess text
def preprocess_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to compute cosine similarity scores
def compute_cosine_similarity_scores(query, retrieved_docs):
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(retrieved_docs, convert_to_tensor=True)
    cosine_scores = np.dot(doc_embeddings, query_embedding.T)
    readable_scores = [{"doc": doc, "score": float(score)} for doc, score in zip(retrieved_docs, cosine_scores.flatten())]
    return readable_scores


profile_id = "data_points"
# Function to answer query with similarity
@app.route('/answer_query', methods=['POST'])
def answer_query():
    try:
        # Extract PDF text from request
        pdf_path = request.json.get('pdf_path')
        extracted_text = extract_text_from_pdf(pdf_path)

        # Preprocess text
        preprocessed_text = preprocess_text(extracted_text)

        # Split text
        embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(preprocessed_text)

        # Create vector store
        vector_store = Chroma.from_texts(texts, embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="stores/insurance_cosine")

        # Load vector store
        load_vector_store = Chroma(persist_directory="stores/insurance_cosine", embedding_function=embeddings)

        # Answer query with similarity
        query = request.json.get('query')
        response = answer_query_with_similarity(query, profile_id)

        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def answer_query_with_similarity(query, profile_id):
    try:
        db3 = Chroma(persist_directory=f"stores/insurance_cosine", embedding_function=embeddings)
        docs = db3.similarity_search(query)
        print(f"\n\nDocuments retrieved: {len(docs)}")
        if not docs:
            print("No documents match the query.")
            return None

        docs_content = [doc.page_content for doc in docs]
        for i, content in enumerate(docs_content, start=1):
            print(f"\nDocument {i}: {content}...")

        cosine_similarity_scores = compute_cosine_similarity_scores(query, docs_content)
        for score in cosine_similarity_scores:
            print(f"\nDocument Score: {score['score']}")

        all_docs_content = " ".join(docs_content)

        template = """
                ### [INST] Instruction:Analyze the provided PDF documents focusing specifically on extracting factual content, mathematical data, and crucial information relevant to device specifications, including discription. Utilize the RAG model's retrieval capabilities to ensure accuracy and minimize the risk of hallucinations in the generated content. Present the findings in a structured and clear format, incorporating:

                    Device Specifications: List all relevant device specifications, including batch numbers, ensuring accuracy and attention to detail.
                    Mathematical Calculations: Perform and report any necessary mathematical calculations found within the PDFs, providing step-by-step explanations to ensure clarity.
                    Numerical Data Analysis: Extract and analyze numerical data from tables included in the documents, summarizing key findings and implications.
                    Factual Information: Highlight crucial factual information extracted from the text, ensuring it is presented in a straightforward and understandable manner.
                    Ensure the response is well-organized, using bullet points or numbered lists where applicable, to enhance readability and presentation. Avoid any form of hallucination by cross-referencing facts with the PDF content directly.

                ### Docs : {docs}
                ### Question : {question}
                """
        prompt = PromptTemplate.from_template(template.format(docs=all_docs_content, question=query))

        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1, token=HUGGINGFACEHUB_API_TOKEN,
                                  top_p=0.15,
                                  max_new_tokens=512,
                                  repetition_penalty=1.1
                                  )
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        answer = llm_chain.run(question=query)
        cleaned_answer = answer.split("Answer:")[-1].strip()

        return cleaned_answer,
    except Exception as e:
        print("An error occurred to get the answer: ", str(e))
        return None

if __name__ == '__main__':
    app.run(debug=True)
