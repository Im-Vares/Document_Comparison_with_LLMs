import streamlit as st
import os
from docx import Document
from huggingface_hub import InferenceClient

# =========================
# Модели Hugging Face API
# =========================
MODEL_OPTIONS = {
    "LLaMA‑3‑8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Mistral‑7B": "mistralai/Mistral-7B-Instruct-v0.2",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "Qwen3": "Qwen/Qwen3-Coder-480B-A35B-Instruct"
}

# =========================
# Hugging Face API Client
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("HF_TOKEN не найден. Установите его: export HF_TOKEN=your_token_here")
else:
    client = InferenceClient(api_key=HF_TOKEN)

# =========================
# Чтение DOCX
# =========================
def read_docx(uploaded_file):
    """Читает DOCX из загруженного файла."""
    try:
        doc = Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        return f"Ошибка чтения DOCX: {e}"

# =========================
# Вызов Hugging Face API
# =========================
def ask_hf(model, prompt):
    """Отправляет запрос в Hugging Face API и возвращает ответ."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2500,
            temperature=0.3
        )
        return completion.choices[0].message["content"]
    except Exception as e:
        return f"Ошибка Hugging Face API: {e}"

# =========================
# Streamlit UI
# =========================
st.title("Сравнение документов через Hugging Face API")

# Выбор модели
model_name = st.selectbox("Выберите модель:", list(MODEL_OPTIONS.keys()), index=0)

# Загрузка документов
col1, col2 = st.columns(2)
with col1:
    uploaded_template = st.file_uploader("Загрузите эталонный DOCX", type=["docx"])
with col2:
    uploaded_example = st.file_uploader("Загрузите проверяемый DOCX", type=["docx"])

# Чтение документов
template_text = read_docx(uploaded_template) if uploaded_template else ""
example_text = read_docx(uploaded_example) if uploaded_example else ""

# Кнопка анализа
if st.button("Сравнить документы"):
    if not template_text or not example_text:
        st.error("Загрузите оба документа для сравнения.")
    else:
        prompt = f"""
Сравни два документа: эталонный и проверяемый.

Выведи строго три раздела:

ОТСУТСТВУЕТ:
- что есть в эталоне, но отсутствует во втором документе

ИЗМЕНЕНО:
- что было → что стало

ДОБАВЛЕНО:
- что есть во втором документе, но отсутствует в эталоне

Не добавляй советы, код или лишние пояснения.
---
Эталон:
{template_text}

---
Проверяемый:
{example_text}

---
Ответ:
"""
        with st.spinner("Выполняется сравнение..."):
            report = ask_hf(MODEL_OPTIONS[model_name], prompt)
        st.subheader("Отчет LLM")
        st.markdown(report)