import streamlit as st
import os
from io import BytesIO
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
    """Отправка запроса в Hugging Face API с обработкой ошибок."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.3
        )
        return completion.choices[0].message["content"]
    except Exception as e:
        return f"Ошибка Hugging Face API: {e}"

def generate_questions(template_text, model):
    """Генерирует проверочные вопросы по эталонному тексту."""
    prompt = f"""Составь 15+ проверочных вопросов для проверки прототипа по эталону.
Вопросы должны охватывать:
- наличие обязательных реквизитов (адрес, даты, сроки действия, компания, должности);
- соответствие формулировок;
- наличие всех таблиц и колонок (ФИО, должность, дата рождения, паспортные данные);
- правильность оформления (подпись, структура документа).
Эталонный текст:
{template_text}
"""
    return ask_hf(model, prompt)

def check_prototype_with_questions(questions_text, example_text, model):
    """Проверяет прототип по списку вопросов и собирает все ответы."""
    try:
        prompt = f"""Отвечай на каждый вопрос на основе текста прототипа.
Для каждого вопроса:
- пиши «✅ есть» или «❌ нет» в начале ответа,
- дай короткое объяснение (1-2 предложения).

Прототип:
{example_text}

Вопросы:
{questions_text}

Формат ответа: нумерованный список с пометками и пояснениями."""
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        return completion.choices[0].message.get("content", "")
    except Exception as e:
        return f"Ошибка проверки вопросов: {e}"

def build_final_report(answers_text, model):
    """Формирует итоговый отчет на основе ответов на вопросы."""
    try:
        prompt = f"""На основе ответов по вопросам составь отчет с инструкцией по исправлению прототипа.
Раздели отчет на три блока: ЧТО ДОБАВИТЬ, ЧТО ИЗМЕНИТЬ, ЧТО УБРАТЬ.
Для каждого пункта:
- кратко объясни, зачем это нужно;
- предложи пример текста.

Ответы по вопросам:
{answers_text}

Формат: списки под каждым блоком."""
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
            temperature=0.3
        )
        return completion.choices[0].message.get("content", "")
    except Exception as e:
        return f"Ошибка формирования отчета: {e}"

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

# Генерация вопросов
if template_text:
    if st.button("Сгенерировать вопросы по эталону"):
        with st.spinner("Генерирую вопросы..."):
            questions = generate_questions(template_text, MODEL_OPTIONS[model_name])
        st.session_state["questions"] = questions
        st.subheader("Вопросы для проверки прототипа")
        st.markdown(questions)

# Проверка прототипа
if "questions" in st.session_state and example_text:
    if st.button("Проверить прототип по вопросам"):
        with st.spinner("Отвечаю на вопросы..."):
            answers = check_prototype_with_questions(st.session_state["questions"], example_text, MODEL_OPTIONS[model_name])
        st.session_state["answers"] = answers
        st.subheader("Ответы по вопросам")
        st.markdown(answers)

# Формирование отчета
if "answers" in st.session_state:
    if st.button("Сформировать итоговый отчет"):
        with st.spinner("Формирую отчет..."):
            final_report = build_final_report(st.session_state["answers"], MODEL_OPTIONS[model_name])
        st.subheader("Итоговый отчет")
        st.markdown(final_report)