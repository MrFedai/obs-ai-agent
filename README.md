# 🎓 OBS AI Agent (Fakülte Asistanı)

Bu proje, fakülteye ait verileri (Yemek listesi, yönetmelikler, akademik takvim vb.) RAG (Retrieval-Augmented Generation) mimarisi ile işleyen ve öğrencilerin sorularını yanıtlayan yerel bir yapay zeka asistanıdır.

## 🚀 Kullanılan Teknolojiler
- **Model:** Llama 3.1 (8B) & Ollama
- **Vektör Veritabanı:** ChromaDB
- **Arayüz:** Streamlit
- **Dil:** Python

## 💻 Kurulum (Lokal)

Proje Arch Linux üzerinde RTX 4070 GPU ile test edilmiştir.
## 💻 Kurulum ve Çalıştırma Rehberi

Bu projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları terminalde sırasıyla uygulayın:

```bash
# 1. Projeyi Git kullanarak bilgisayarımıza çekiyoruz
git clone [https://github.com/MrFeda/obs-ai-agent.git](https://github.com/MrFeda/obs-ai-agent.git)
cd obs-ai-agent

# 2. Sanal ortamı (Virtual Environment) oluşturuyoruz
python -m venv venv

# 3. Sanal ortamı aktif ediyoruz
source venv/bin/activate

# 4. Gerekli kütüphaneleri yüklüyoruz
pip install -r requirements.txt

# 5. Ollama modellerini bir seferliğine kuruyoruz (Llama 3.1 ve Embedding modeli)
ollama pull llama3.1
ollama pull nomic-embed-text

# 6. Veri klasörünü oluşturuyoruz
mkdir data
# ÖNEMLİ: Bu aşamada analiz edilecek PDF dosyalarını (Yemek listesi, yönetmelik vb.)
# dosya yöneticisinden açıp oluşturduğunuz 'data' klasörünün içine kopyalayın.

# 7. Veritabanını güncelliyoruz (PDF'leri okuyip vektöre çevirir)
python ingest.py

# 8. Projeyi lokal olarak çalıştırıyoruz
streamlit run app.py
