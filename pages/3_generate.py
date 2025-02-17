import os
import random
import re
import uuid
import streamlit as st
import fitz  # PyMuPDF
import PyPDF2
import google.generativeai as genai
import PIL.Image
from langchain.chains.llm import LLMChain
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import StopCandidateException
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationChain
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="AskMe", layout="wide")
TEMP_DIR = "temp"

# --------------------- Global CSS ---------------------
global_css = """
<style>
    body { 
        background-color: #F4F4F9; 
        font-family: 'Roboto', sans-serif; 
    }
    /* Override default buttons */
    button {
        background-color: #4CAF50 !important;
        color: white !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 6px !important;
        font-size: 16px !important;
        cursor: pointer !important;
        transition: background-color 0.3s !important;
    }
    button:hover {
        background-color: #45A049 !important;
    }
    /* Style for search input */
    .css-1aumxhk input {
        border: 2px solid #4CAF50 !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-size: 16px !important;
    }
</style>
"""
st.markdown(global_css, unsafe_allow_html=True)

# --------------------- Session State Initialization ---------------------
if 'fulltext' not in st.session_state:
    st.session_state['fulltext'] = ""
if 'question' not in st.session_state:
    st.session_state['question'] = ""
if 'generate_question' not in st.session_state:
    st.session_state.generate_question = False
if 'generated' not in st.session_state:
    st.session_state.generated = False
if 'generated_hard' not in st.session_state:
    st.session_state.generated_hard = False
if 'generated_easy' not in st.session_state:
    st.session_state.generated_easy = False
if 'answered' not in st.session_state:
    st.session_state['answered'] = False
if 'selected_answer' not in st.session_state:
    st.session_state['selected_answer'] = ''
if 'is_correct' not in st.session_state:
    st.session_state['is_correct'] = False
# Persistent unique key for the current question (used for answer button keys)
if 'current_question_uuid' not in st.session_state:
    st.session_state.current_question_uuid = str(uuid.uuid4())
# Quiz state for Quiz Mode
if 'quiz_questions' not in st.session_state:
    st.session_state.quiz_questions = []
if 'quiz_index' not in st.session_state:
    st.session_state.quiz_index = 0
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0
if 'quiz_history' not in st.session_state:
    st.session_state.quiz_history = []

# --------------------- Global LLM & Conversation Chain ---------------------
API_KEY = 'AIzaSyBbnLBTj3xqJe3oQxTQtkDo1F6ZW47H8nA'
genai.configure(api_key=API_KEY)

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    convert_system_message_to_human=True,
    google_api_key=API_KEY,
    temperature=0.3
)
st.session_state.llm = llm

if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = ConversationChain(
        llm=llm,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=ConversationEntityMemory(llm=llm)
    )

vision_model = ChatGoogleGenerativeAI(
    model="gemini-pro-vision",
    google_api_key=API_KEY
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=API_KEY
)

memory = ConversationEntityMemory(llm=llm)
conversation = ConversationChain(llm=llm, prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE, memory=memory)

# --------------------- Utility Functions ---------------------
def handle_answer_selection(selected_answer):
    st.session_state['answered'] = True
    st.session_state['selected_answer'] = selected_answer
    correct_answer = st.session_state['correct_answer']
    if selected_answer in correct_answer:
        st.session_state['is_correct'] = True
    else:
        st.session_state['is_correct'] = False

def reformat_output(output):
    output = output.replace("*", "")
    if "Question:" in output:
        q_split = output.split("Question:", 1)[1]
        q_match = re.search(r'(.+?\?)', q_split)
        if q_match:
            question_txt = q_match.group(1).strip()
        else:
            question_txt = q_split.strip()
    else:
        question_txt = output.splitlines()[0].strip()
    answers_txt = re.findall(r'(?:^|\n)([A-D][\).\s].+)', output)
    answers_txt = [ans.strip() for ans in answers_txt]
    correct_letter_match = re.search(r'Correct Answer:\s*([A-D])', output, re.IGNORECASE)
    correct_answer = None
    if correct_letter_match:
        correct_letter = correct_letter_match.group(1).upper()
        for ans in answers_txt:
            if ans.startswith(correct_letter + ")") or ans.startswith(correct_letter + "."):
                correct_answer = ans
                break
    if not correct_answer:
        correct_answer = "No correct answer found"
    if len(answers_txt) > 4:
        if correct_answer not in answers_txt[:4]:
            correct_answer = "No correct answer found"
        answers_txt = answers_txt[:4]
    while len(answers_txt) < 4:
        answers_txt.append(f"{chr(65+len(answers_txt))}) Option {len(answers_txt)+1}")
    return question_txt, answers_txt, correct_answer

def display_question_answer(question, answers, correct_answer):
    custom_css = """
    <style>
        .question-container {
            background-color: #ffffff;
            border-left: 5px solid #4CAF50;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .header { font-weight: 700; font-size: 20px; color: #333333; margin-bottom: 10px; }
        .content { font-size: 18px; color: #555555; }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    if isinstance(answers, list):
        st.session_state['answers'] = answers
    else:
        st.session_state['answers'] = [a.strip() for a in answers.split('\n') if a.strip()[:4]]
    st.session_state['correct_answer'] = correct_answer
    unique_key = st.session_state.current_question_uuid
    st.markdown(
        f"<div class='question-container'><div class='header'>Question</div>"
        f"<div class='content'>{question}</div></div>",
        unsafe_allow_html=True
    )
    for idx, answer in enumerate(st.session_state['answers']):
        if not answer or answer.strip() == f"({chr(65+idx)}) Options:":
            continue
        if st.button(answer, key=f'answer_{idx}_{unique_key}', 
                     help=f"Select option {chr(65+idx)}", 
                     use_container_width=True):
            handle_answer_selection(answer)
    if st.session_state.get('answered'):
        if st.session_state.get('is_correct'):
            st.success("Correct! 🎉")
        else:
            if st.session_state['correct_answer'] == "No correct answer found":
                st.warning("No valid correct answer was found in the model's response.")
            else:
                st.error(f"Incorrect. The correct answer is: {st.session_state['correct_answer']}")
        st.session_state['answered'] = False

def extract_text_from_pdf(doc_path, start_page, end_page):
    pages = []
    with open(doc_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(start_page - 1, end_page):
            if page_num < len(pdf_reader.pages):
                page = pdf_reader.pages[page_num]
                pages.append(page.extract_text())
    return pages

def extract_text_from_photo(image_path, vision_model):
    try:
        img_pil = PIL.Image.open(image_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Save the text shown in the image"},
                {"type": "image_url", "image_url": img_pil}
            ]
        )
        output = vision_model.invoke([message])
        return output.content
    except StopCandidateException:
        return "error"

# --------------------- Main UI with Tabbed Interface ---------------------
st.markdown(
    """
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 32px; color: #333333;">🤔 Let's Expand Your Knowledge</h1>
        <p style="font-size: 18px; color: #555555;">Extract text from your document, search it, and generate custom questions and quizzes.</p>
    </div>
    """, unsafe_allow_html=True
)
st.write("---")

with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center;">
            <h2 style="color: #4CAF50;">AskMe Sidebar</h2>
        </div>
        """, unsafe_allow_html=True
    )
    # Search input and button (using a separate key to avoid conflicts)
    search_query = st.text_input("🔍 Enter your question:", "", key="search_query", help="Type your query here")
    search_button = st.button("Search", key="search_btn")
    st.markdown("<hr>", unsafe_allow_html=True)
    uploaded_files = os.listdir(TEMP_DIR)
    selected_file = st.selectbox("Select a file to analyze:", uploaded_files)

# Document Extraction & Search (Sidebar)
if selected_file:
    file_path = os.path.join(TEMP_DIR, selected_file)
    with st.sidebar:
        if selected_file.lower().endswith(".pdf"):
            doc_path = os.path.join(TEMP_DIR, selected_file)
            pdf_document = fitz.open(file_path)
            st.info("For faster performance, choose a few pages.")
            col1, col2 = st.columns(2)
            with col1:
                start_page = st.number_input("Start page", value=1, min_value=1,
                                             max_value=pdf_document.page_count, step=1)
            with col2:
                end_page = st.number_input("End page", value=pdf_document.page_count, 
                                           min_value=1, max_value=pdf_document.page_count, step=1)
        else:
            image_path = os.path.join(TEMP_DIR, selected_file)
        if st.button("Extract Text", key="extract_text", help="Extract text from the file"):
            st.session_state['display_text'] = True
            if selected_file.lower().endswith(".pdf"):
                pages = extract_text_from_pdf(doc_path, start_page, end_page)
                st.session_state['pdf_pages'] = pages
                st.session_state['fulltext'] = "\n".join(pages)
                if start_page <= end_page:
                    for i, page_text in enumerate(pages, start=start_page):
                        with st.expander(f"Page {i} ({len(page_text)} characters)"):
                            st.write(page_text)
                else:
                    st.warning("End page must be ≥ start page.")
            else:
                fulltext = extract_text_from_photo(file_path, vision_model)
                if fulltext == "error":
                    st.session_state['fulltext'] = ""
                    st.error("Failed to extract text from the image. Try a different image.", icon="🚨")
                else:
                    st.session_state['fulltext'] = fulltext
                    with st.expander(f"Extracted Text ({len(fulltext)} characters)"):
                        st.write(fulltext)
        else:
            if st.session_state.get('display_text', False):
                if selected_file.lower().endswith(".pdf") and 'pdf_pages' in st.session_state:
                    for i, page_text in enumerate(st.session_state['pdf_pages'], start=start_page):
                        with st.expander(f"Page {i} ({len(page_text)} characters)"):
                            st.write(page_text)
                else:
                    fulltext = st.session_state.get('fulltext', "")
                    if fulltext:
                        with st.expander(f"Extracted Text ({len(fulltext)} characters)"):
                            st.write(fulltext)
    with st.sidebar:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Document Search Results")
    if search_button:
        fulltext = st.session_state.get('fulltext')
        question_search = st.session_state.get('search_query')
        if question_search.strip() and fulltext:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            texts = text_splitter.split_text(fulltext)
            vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 10})
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=vector_index,
                return_source_documents=True
            )
            result = qa_chain({"query": question_search})
            document_answer = result["result"]
            template_gemini = """Please answer the following question:
{question}
"""
            prompt_gemini = ChatPromptTemplate.from_template(template_gemini)
            chain_gemini = LLMChain(llm=llm, prompt=prompt_gemini)
            gemini_answer = chain_gemini.invoke({"question": question_search})['text']
            st.markdown("*Answer based on Document:*")
            with st.expander(f"Document Answer ({len(document_answer)} characters)"):
                st.markdown(document_answer)
            st.markdown("*Gemini Answer:*")
            with st.expander(f"Gemini Answer ({len(gemini_answer)} characters)"):
                st.markdown(gemini_answer)

# --------------------- Main Page Tabs ---------------------
tab_generate, tab_quiz = st.tabs(["Generate Question", "Quiz Mode"])

# ----- Tab: Generate Question -----
with tab_generate:
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;">
            <h2 style="color: #333;">Generate a New Question</h2>
        </div>
        """, unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns(3)
    
    # Generate New Question
    with col1:
        Clicked = st.button("Generate New Question", key="new_question", help="Click to generate a new question")
        if Clicked:
            if st.session_state['fulltext'] == "":
                st.warning("Please extract text first.")
            else:
                fulltext = st.session_state.get('fulltext', "")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = text_splitter.split_text(fulltext)
                context_text = random.choice(chunks) if chunks else fulltext
                template_new = """
                You are a helpful AI. Please create a novel multiple-choice question from the text below.
                The final output must follow this exact format, with no extra lines or explanations:

                Question: <question text>
                A) <answer text>
                B) <answer text>
                C) <answer text>
                D) <answer text>
                Correct Answer: <A or B or C or D>

                No additional text, disclaimers, or explanations.

                Text to base the question on:
                {context_text}
                """
                prompt_new = ChatPromptTemplate.from_template(template_new)
                chain_new = LLMChain(llm=llm, prompt=prompt_new)
                output = chain_new.invoke({"context_text": context_text})['text']
                # No debug line, so nothing displayed
                q, a, c = reformat_output(output)
                st.session_state['question_output'] = q
                st.session_state['answers'] = a
                st.session_state['correct_answer'] = c
                st.session_state.generate_question = True
                st.session_state.generated = True
                st.session_state.current_question_uuid = str(uuid.uuid4())
        if st.session_state.get('generated', False):
            q = st.session_state['question_output']
            a = st.session_state['answers']
            c = st.session_state['correct_answer']
            display_question_answer(q, a, c)
    
    # Generate Harder Question
    with col2:
        if st.session_state.generate_question:
            Clicked_Harder = st.button("Generate Harder Question", key="hard_question", help="Click to generate a harder question")
            if Clicked_Harder:
                st.session_state.generated = False
                st.session_state.generated_hard = True
                st.session_state.generated_easy = False
                prev_q = st.session_state.get('question_output', "")
                prev_c = st.session_state.get('correct_answer', "")
                fulltext = st.session_state.get('fulltext', "")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = text_splitter.split_text(fulltext)
                context_text = random.choice(chunks) if chunks else fulltext
                input_text = (
                    f"Generate a novel multiple-choice question that is more difficult than the previous one. "
                    f"Previous question was: {prev_q} and the previous correct answer was: {prev_c}. "
                    f"Use the following context: {context_text}\n\n"
                    "Format:\n"
                    "Question: <question text>\n"
                    "A) <answer text>\n"
                    "B) <answer text>\n"
                    "C) <answer text>\n"
                    "D) <answer text>\n"
                    "Correct Answer: <A or B or C or D>\n"
                )
                output_harder = st.session_state.conversation_chain.predict(input=input_text)
                q, a, c = reformat_output(output_harder)
                st.session_state['question_output'] = q
                st.session_state['answers'] = a
                st.session_state['correct_answer'] = c
                st.session_state.current_question_uuid = str(uuid.uuid4())
            if st.session_state.get('generated_hard', False):
                q = st.session_state['question_output']
                a = st.session_state['answers']
                c = st.session_state['correct_answer']
                display_question_answer(q, a, c)
    
    # Generate Easier Question
    with col3:
        if st.session_state.generate_question:
            Easier_Clicked = st.button("Generate Easier Question", key="easy_question", help="Click to generate an easier question")
            if Easier_Clicked:
                st.session_state.generated = False
                st.session_state.generated_hard = False
                st.session_state.generated_easy = True
                prev_q = st.session_state.get('question_output', "")
                prev_c = st.session_state.get('correct_answer', "")
                fulltext = st.session_state.get('fulltext', "")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = text_splitter.split_text(fulltext)
                context_text = random.choice(chunks) if chunks else fulltext
                input_text = (
                    f"Generate a novel multiple-choice question that is easier than the previous one. "
                    f"Previous question was: {prev_q} and the previous correct answer was: {prev_c}. "
                    f"Use the following context: {context_text}\n\n"
                    "Format:\n"
                    "Question: <question text>\n"
                    "A) <answer text>\n"
                    "B) <answer text>\n"
                    "C) <answer text>\n"
                    "D) <answer text>\n"
                    "Correct Answer: <A or B or C or D>\n"
                )
                output_easier = st.session_state.conversation_chain.predict(input=input_text)
                q, a, c = reformat_output(output_easier)
                st.session_state['question_output'] = q
                st.session_state['answers'] = a
                st.session_state['correct_answer'] = c
                st.session_state.current_question_uuid = str(uuid.uuid4())
            if st.session_state.get('generated_easy', False):
                q = st.session_state['question_output']
                a = st.session_state['answers']
                c = st.session_state['correct_answer']
                display_question_answer(q, a, c)

# ----- Tab: Quiz Mode -----
tab_quiz = tab_quiz
with tab_quiz:
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;">
            <h2 style="color: #333;">Document-Based Quiz</h2>
            <p style="font-size: 16px; color: #555;">Answer 10 multiple-choice questions generated from your document.</p>
        </div>
        """, unsafe_allow_html=True
    )
    if st.button("Start Quiz / Restart Quiz", key="start_quiz"):
        st.session_state.quiz_questions = []
        st.session_state.quiz_index = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_history = []
        if st.session_state['fulltext'] != "":
            fulltext = st.session_state['fulltext']
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            chunks = text_splitter.split_text(fulltext)
            for i in range(10):
                context_text = random.choice(chunks) if chunks else fulltext
                template_quiz = """
                You are a helpful AI. Please create a multiple-choice question (with 4 options) based on the text below.
                The final output must follow this exact format (with no extra text):

                Question: <question text>
                A) <answer text>
                B) <answer text>
                C) <answer text>
                D) <answer text>
                Correct Answer: <A or B or C or D>
                Text to base the question on:
                {context_text}
                """
                prompt_quiz = ChatPromptTemplate.from_template(template_quiz)
                chain_quiz = LLMChain(llm=llm, prompt=prompt_quiz)
                output_quiz = chain_quiz.invoke({"context_text": context_text})['text']
                q, a, c = reformat_output(output_quiz)
                st.session_state.quiz_questions.append({"question": q, "options": a, "correct": c})
        else:
            st.warning("Please extract text first to generate the quiz.")
    
    if st.session_state.quiz_questions:
        if st.session_state.quiz_index < len(st.session_state.quiz_questions):
            current = st.session_state.quiz_index
            total = len(st.session_state.quiz_questions)
            st.markdown(f"**Question {current+1} of {total}:**")
            current_q = st.session_state.quiz_questions[current]
            st.markdown(f"**{current_q['question']}**")
            option_cols = st.columns(2)
            for i, option in enumerate(current_q["options"]):
                if option_cols[i % 2].button(option, key=f"quiz_q{current}_opt{i}"):
                    if option == current_q["correct"]:
                        st.session_state.quiz_score += 1
                        st.success("Correct!")
                        st.session_state.quiz_history.append({
                            "question": current_q["question"],
                            "selected": option,
                            "result": "Correct"
                        })
                    else:
                        st.error(f"Incorrect. The correct answer is: {current_q['correct']}")
                        st.session_state.quiz_history.append({
                            "question": current_q["question"],
                            "selected": option,
                            "result": "Incorrect"
                        })
                    st.session_state.quiz_index += 1
                    st.experimental_rerun()
            progress = int((st.session_state.quiz_index / total) * 100)
            st.progress(progress)
            st.markdown(f"Score: **{st.session_state.quiz_score} / {total}**")
        else:
            st.header("Quiz Complete!")
            total = len(st.session_state.quiz_questions)
            st.markdown(f"Your final score is **{st.session_state.quiz_score}** out of **{total}**")
            st.markdown("### Quiz History")
            for record in st.session_state.quiz_history:
                st.markdown(f"- **Q:** {record['question']}\n   **Your answer:** {record['selected']} — *{record['result']}*")
            if st.button("Restart Quiz", key="restart_quiz"):
                st.session_state.quiz_questions = []
                st.session_state.quiz_index = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_history = []
                st.experimental_rerun()
    else:
        st.info("Start the quiz by clicking the 'Start Quiz / Restart Quiz' button.")

st.write("---")
