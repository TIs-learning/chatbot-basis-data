import streamlit as st
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load environment variables (API Key dari .env)
load_dotenv()
import os

# 2. Setup Client LLM (Sesuai yang lu pakai sebelumnya)
llm_client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1/"
) 

# 3. Setup ChromaDB (Panggil folder database buku lu DENGAN CACHE)
@st.cache_resource
def load_database():
    chroma_client = chromadb.PersistentClient(path="./buku_pengantar_basis_data_full")
    return chroma_client.get_collection(name="pengantar_basis_data_full")

collection = load_database()
# 4. Tes Tampilan UI Streamlit
st.title("Chatbot Pengantar Basis Data 🤖")
st.write("Status: Database dan LLM berhasil disambungkan!")

# ... (kodingan setup llm_client dan chroma_client lu biarin aja di atas) ...

st.title("Chatbot Pengantar Basis Data 🤖")

# 1. Masukin fungsi RAG lu yang udah mateng dari Modul 5
def tanya_dokumen(pertanyaan_user):
    hasil_pencarian = collection.query(
        query_texts=[pertanyaan_user],
        n_results=3 
    )
    konteks_gabungan = "\n\n---\n\n".join(hasil_pencarian['documents'][0])
    
    # Pake prompt ketat lu biar AI-nya to the point
    prompt_final = f"""Anda adalah asisten AI yang sangat ketat, kaku, dan langsung pada intinya. 
ATURAN MENJAWAB:
1. Jawab HANYA berdasarkan konteks di bawah ini. Jika tidak ada, jawab "Saya tidak tahu dari dokumen yang ada".
2. Jawablah HANYA sesuai apa yang ditanyakan dan sangat singkat.

Konteks:
{konteks_gabungan}

Pertanyaan:
{pertanyaan_user}"""

    # Sesuaikan dengan model andalan lu
    response = llm_client.chat.completions.create(
        model="openai/gpt-oss-120b", 
        messages=[{"role": "user", "content": prompt_final}]
    )
    return response.choices[0].message.content


# 2. Setup Memori Chat (Session State) biar chat nggak hilang
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Tampilkan riwayat chat lama di layar
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Bikin kotak input chat di bawah layar
if prompt := st.chat_input("Tanyakan sesuatu tentang Basis Data..."):
    
    # Tampilkan chat user di layar & simpan ke memori
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Panggil fungsi RAG & tampilkan jawaban AI di layar
    with st.chat_message("assistant"):
        jawaban_ai = tanya_dokumen(prompt)
        st.markdown(jawaban_ai)
    
    # Simpan jawaban AI ke memori
    st.session_state.messages.append({"role": "assistant", "content": jawaban_ai})
