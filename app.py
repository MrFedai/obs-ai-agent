import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Sayfa Ayarları
st.set_page_config(page_title="OBS AI Asistan", layout="wide") # Layout'u wide yaptım ki debug paneli rahat görünsün
st.title("🎓 Fakülte AI Asistanı")

# Model ve DB Kurulumu
@st.cache_resource
def init_rag():
    # 1. Embedding ve Veritabanı Bağlantısı
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
    
    # 2. Retriever (Getirici) Ayarı - En alakalı 5 parça
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) 
    
    # 3. LLM (Beyin) Ayarı
    llm = ChatOllama(model="llama3.1", temperature=0) # Halüsinasyonu önlemek için 0
    
    # 4. Prompt Şablonu
    template = """
    Sen üniversite öğrencilerinin sorularını yanıtlayan yardımsever bir asistanısın.
    Aşağıda verilen bağlam (Context) bilgilerini kullanarak soruyu Türkçe cevapla.
    
    Eğer bağlamda sorunun cevabı yoksa, dürüstçe "Verilen dökümanlarda bu bilgi yer almıyor" de.
    Uydurma cevap verme.
    
    Bağlam: {context}
    
    Soru: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    return retriever, prompt, llm

# Sistemi başlat
retriever, prompt, llm = init_rag()

# --- SOHTBET GEÇMİŞİ BAŞLATMA (HATA BURADAYDI) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajları ekrana çiz
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- YENİ MESAJ MANTIĞI ---
if prompt_input := st.chat_input("Sorunuzu yazın..."):
    # 1. Kullanıcı mesajını ekle ve göster
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # 2. AI Cevabını Üret
    with st.chat_message("assistant"):
        status_container = st.status("Dökümanlar taranıyor...", expanded=False)
        
        try:
            # A. Dökümanları Bul
            docs = retriever.invoke(prompt_input)
            
            # B. Yan Panele (Sidebar) Kanıtları Yazdır (DEBUG MODU)
            with st.sidebar:
                st.header("🔍 Modelin Gözü")
                st.write(f"**Soru:** {prompt_input}")
                st.divider()
                if not docs:
                    st.error("❌ Veritabanında alakalı kayıt bulunamadı.")
                
                for i, doc in enumerate(docs):
                    with st.expander(f"📄 Kanıt {i+1} (Kaynak: {doc.metadata.get('source', 'Bilinmiyor')})"):
                        st.caption(f"Sayfa: {doc.metadata.get('page', '-')}")
                        st.info(doc.page_content) # İçeriği göster
            
            # C. Cevabı Üret
            context_text = "\n\n".join([d.page_content for d in docs])
            chain = prompt | llm | StrOutputParser()
            
            response = chain.invoke({"context": context_text, "question": prompt_input})
            
            status_container.update(label="Cevap hazır!", state="complete", expanded=False)
            st.markdown(response)
            
            # Cevabı geçmişe kaydet
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")
