from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------------------
# Load Embedding Model
# ---------------------------
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ---------------------------
# Create Chroma DB
# ---------------------------
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="studymate")

# ---------------------------
# Chunk text
# ---------------------------
def split_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# ---------------------------
# Add PDF text to Vector DB
# ---------------------------
def add_to_vector_db(text):
    chunks = split_text(text)
    for i, chunk in enumerate(chunks):
        embedding = embedder.encode([chunk])[0].tolist()
        collection.add(
            ids=[str(i)],
            documents=[chunk],
            embeddings=[embedding]
        )

# ---------------------------
# Load AI Model (small)
# ---------------------------
model_name = "ibm-granite/granite-3.2-2b-instruct"        # small + works locally
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto"
)

# ---------------------------
# Answering
# ---------------------------
def answer_question(question, top_k=3):
    # Search best chunks
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    context = "\n".join(results["documents"][0])

    prompt = f"""You are StudyMate.
Answer the question only using this context:

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.2
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("ANSWER:")[-1].strip()
