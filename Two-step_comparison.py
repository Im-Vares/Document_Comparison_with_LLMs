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
    "Qwen2.5‑7B": "Qwen/Qwen2.5-7B-Instruct",
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
    """Читает текст из DOCX."""
    try:
        doc = Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        return f"Ошибка чтения DOCX: {e}"

# =========================
# Вызов Hugging Face API
# =========================
def ask_hf(model, prompt):
    """Отправка запроса в Hugging Face API и получение ответа."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
            temperature=0.3
        )
        return completion.choices[0].message["content"]
    except Exception as e:
        return f"Ошибка Hugging Face API: {e}"

# =========================
# Streamlit UI
# =========================
st.title("Сравнение документов")

# Выбор модели
model_name = st.selectbox("Выберите модель:", list(MODEL_OPTIONS.keys()), index=0)

# Загрузка документов
col1, col2 = st.columns(2)
with col1:
    uploaded_template = st.file_uploader("Загрузите шаблонный документ (DOCX)", type=["docx"])
with col2:
    uploaded_example = st.file_uploader("Загрузите проверяемый документ (DOCX)", type=["docx"])

# Чтение документов
template_text = read_docx(uploaded_template) if uploaded_template else ""
example_text = read_docx(uploaded_example) if uploaded_example else ""

# Кнопка анализа
if st.button("Сравнить документы"):
    if not template_text or not example_text:
        st.error("Загрузите оба документа для анализа.")
    else:
        prompt = f"""
Ты – аудитор документов. Сравни шаблон (эталон) и проверяемый вариант.

Сделай два шага анализа:

1. Template → Prototype
   - Определи, какие элементы шаблона полностью отсутствуют (пометь как 'Отсутствует').
   - Определи, какие элементы присутствуют, но частично (пометь как 'Частично').

2. Prototype → Template
   - Найди, что добавлено в проверяемый документ, но отсутствует в шаблоне (пометь как 'Лишнее').

Ответ должен содержать три раздела:

ОТСУТСТВУЕТ:
- ...

ЧАСТИЧНО:
- ...

ЛИШНЕЕ:
- ...

---
ШАБЛОН:
{template_text}

---
ПРОВЕРЯЕМЫЙ:
{example_text}

---
Ответ:
"""
        with st.spinner("Выполняется анализ..."):
            report = ask_hf(MODEL_OPTIONS[model_name], prompt)
        st.subheader("Отчет сравнения")
        st.markdown(report)