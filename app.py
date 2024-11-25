import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from pinecone import Pinecone, ServerlessSpec, Index

# Initialize Pinecone instance
pc = Pinecone(api_key="pcsk_4bHEfS_37sPFUVPwFjP2vdB59Ginij3LLXpShnLesezTJ13U6F3u7SbB4UgpZ5H3F1VEuG")

# Check if the index exists, and create it if not
index_name = "medical-knowledge-base"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=768,  # Adjust based on your embedding size
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the Pinecone index
index = Index(index_name, host='https://medical-knowledge-base-f393qzo.svc.aped-4627-b74a.pinecone.io') 

# Initialize Generator Model (using distilgpt2 here as an example)
model_name = "distilgpt2"  # Using a smaller model
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator_model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to Retrieve Similar Chunks
def retrieve_similar(query, top_k=5):
    query_embedding = model_name.encode(query)  # Replace with your embedding logic
    results = index.query(query_embedding, top_k=top_k, include_metadata=True)
    return [result["metadata"]["combined"] for result in results["matches"]]

# Function to Generate Response
def generate_response(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    prompt = f"""
    You are a medical assistant chatbot. Answer the user's query based on the following context:

    Context:
    {context}

    Query: {query}

    Response:
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = generator_model.generate(inputs["input_ids"], max_length=200, num_beams=5, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Streamlit App
st.title("RAG Medical Chatbot")
st.write("Ask questions about diseases, symptoms, and treatments!")

user_query = st.text_input("Enter your medical query:")

if st.button("Get Response"):
    if user_query.strip():
        with st.spinner("Generating response..."):
            retrieved_chunks = retrieve_similar(user_query)
            if retrieved_chunks:
                response = generate_response(user_query, retrieved_chunks)
                st.success("Chatbot Response:")
                st.write(response)
            else:
                st.error("No relevant information found!")
    else:
        st.error("Please enter a query.")
