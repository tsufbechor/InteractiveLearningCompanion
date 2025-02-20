import os
import random
import re
import uuid
import tempfile
import streamlit as st
import fitz  # PyMuPDF
import PyPDF2
import requests
import google.generativeai as genai
import PIL.Image
from urllib.parse import quote_plus
from newspaper import Article, Config
from bs4 import BeautifulSoup
from langchain.chains.llm import LLMChain
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import StopCandidateException
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationChain, RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --------------------- Page Config & Global CSS ---------------------
st.set_page_config(page_title="AskMe", layout="wide")
TEMP_DIR = "temp"

global_css = """
<style>
    body { 
        background-color: #F4F4F9; 
        font-family: 'Roboto', sans-serif; 
    }
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
if 'domain' not in st.session_state:
    st.session_state['domain'] = ""
# articles_text is a list of article dictionaries
if 'articles_text' not in st.session_state:
    st.session_state['articles_text'] = []
if 'knowledge_base' not in st.session_state:
    st.session_state['knowledge_base'] = None

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

# --------------------- Global LLM & Related Chains ---------------------
API_KEY = 'ENTER YOUR API KEY'
genai.configure(api_key = API_KEY)

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

# --------------------- Utility Functions for Domain & KB ---------------------
def extract_text_from_pdf(file_path, start_page, end_page):
    pages = []
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(start_page - 1, end_page):
            if page_num < len(pdf_reader.pages):
                pages.append(pdf_reader.pages[page_num].extract_text())
    return "\n".join(pages)

def extract_text_from_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.error(f"Error reading TXT file: {e}")
        return ""

def determine_document_domain(document_text):
    prompt = f"""
Based on the following document text, determine its primary domain or subject.
Provide a concise one-word or short-phrase answer (e.g., "Classic/Modern ML", "Sports", "Economics", etc.).

Document Text:
\"\"\"{document_text[:1000]}...\"\"\"

Answer:"""
    domain = st.session_state.llm.predict(prompt)
    return domain.strip()

def get_article_content(url):
    """
    Retrieve full article content using newspaper3k with a custom Config
    that mimics a real browser. If the content is too short, fallback to BeautifulSoup.
    """
    try:
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
        config = Config()
        config.browser_user_agent = user_agent
        config.request_timeout = 15
        config.http_headers = {
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/'
        }
        article = Article(url, config=config)
        article.download()
        article.parse()
        text = article.text.strip()
        if len(text) < 200:
            response = requests.get(url, headers=config.http_headers, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'lxml')
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text(separator="\n").strip()
        return text
    except Exception as e:
        st.warning(f"Failed to retrieve article from {url}: {e}")
        return ""

def get_wikipedia_full_text(title):
    """
    Retrieve the full text of a Wikipedia article using the parse API.
    This should return the entire page content.
    """
    url = f"https://en.wikipedia.org/w/api.php?action=parse&page={quote_plus(title)}&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        html = data.get("parse", {}).get("text", {}).get("*", "")
        soup = BeautifulSoup(html, "lxml")
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator="\n").strip()
        return text
    return ""

def search_web_for_articles(domain):
    """
    Use Wikipedia's API to search for articles related to the given domain.
    Returns a list (up to 10) of dictionaries with keys: title, link, content.
    """
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f"{domain} research",
        "format": "json"
    }
    response = requests.get(search_url, params=params)
    articles = []
    if response.status_code == 200:
        data = response.json()
        search_results = data.get("query", {}).get("search", [])[:5]
        for result in search_results:
            title = result.get("title", "")
            full_text = get_wikipedia_full_text(title)
            link = f"https://en.wikipedia.org/wiki/{quote_plus(title)}"
            articles.append({
                "title": title,
                "link": link,
                "content": full_text
            })
        return articles
    else:
        st.error("Failed to retrieve Wikipedia articles.")
        return []

def create_knowledge_base(document_text, articles_text):
    """
    Build a knowledge base from the document text combined with external article content.
    articles_text is expected to be a list of article dictionaries.
    """
    combined_text = document_text
    if articles_text:
        wiki_text = "\n".join([article["content"] for article in articles_text])
        combined_text = document_text + "\n" + wiki_text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(combined_text)
    vector_store = Chroma.from_texts(chunks, embeddings, collection_name="combined_kb")
    return vector_store

# --------------------- Existing Utility Functions ---------------------
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

def generate_chain_of_thought_explanation(question, options, correct_answer, selected_answer):
    options_str = "\n".join(options)
    explanation_prompt = f"""
You are a knowledgeable AI tutor. Provide a detailed chain-of-thought explanation for the following multiple-choice question.
Explain why the correct answer is correct. If the user’s answer is incorrect, explain where they went wrong and key concepts to review.

Question: {question}
Options:
{options_str}
Correct Answer: {correct_answer}
User Selected: {selected_answer}

Explanation:
"""
    explanation = st.session_state.llm.predict(explanation_prompt)
    return explanation

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
        explanation = generate_chain_of_thought_explanation(
            question,
            st.session_state['answers'],
            st.session_state['correct_answer'],
            st.session_state['selected_answer']
        )
        st.markdown("**Explanation:**")
        st.write(explanation)
        st.session_state['answered'] = False

# --------------------- Main UI with Sidebar & Tabs ---------------------
st.markdown(
    """
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 32px; color: #333333;">🤔 Let's Expand Your Knowledge</h1>
        <p style="font-size: 18px; color: #555555;">
        Extract text from your document, then choose whether to enrich your knowledge base with related Wikipedia articles.
        The AskMe Sidebar uses the combined KB (document + Wikipedia), while Generate Question and Quiz are based solely on the document.
        </p>
    </div>
    """, unsafe_allow_html=True
)
st.write("---")

# --------------------- Sidebar: Document Extraction & KB Building ---------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center;">
            <h2 style="color: #4CAF50;">AskMe Sidebar</h2>
        </div>
        """, unsafe_allow_html=True
    )
    search_query = st.text_input("🔍 Enter your question:", "", key="search_query", help="Type your query here")
    search_button = st.button("Search", key="search_btn")
    st.markdown("<hr>", unsafe_allow_html=True)
    if os.path.exists(TEMP_DIR):
        uploaded_files = os.listdir(TEMP_DIR)
    else:
        uploaded_files = []
    selected_file = st.selectbox("Select a file to analyze:", uploaded_files)
    
    # Build KB for AskMe Sidebar using document text + Wikipedia articles (if opted)
    if selected_file:
        file_path = os.path.join(TEMP_DIR, selected_file)
        with st.sidebar:
            if selected_file.lower().endswith(".pdf"):
                pdf_document = fitz.open(file_path)
                st.info("For faster performance, choose a page range.")
                col1, col2 = st.columns(2)
                with col1:
                    start_page = st.number_input("Start page", value=1, min_value=1,
                                                 max_value=pdf_document.page_count, step=1)
                with col2:
                    end_page = st.number_input("End page", value=pdf_document.page_count, 
                                               min_value=1, max_value=pdf_document.page_count, step=1)
            if st.button("Extract Text", key="extract_text", help="Extract text from the file"):
                st.session_state['display_text'] = True
                if selected_file.lower().endswith(".pdf"):
                    fulltext = extract_text_from_pdf(file_path, start_page, end_page)
                else:
                    fulltext = extract_text_from_txt(file_path)
                st.session_state['fulltext'] = fulltext
                if fulltext:
                    with st.expander(f"Extracted Text ({len(fulltext)} characters)"):
                        st.write(fulltext)
                    # Option to enrich KB with external Wikipedia articles
                    use_external = st.checkbox("Enrich KB with external Wikipedia articles", value=True)
                    if use_external:
                        with st.spinner("Determining document domain..."):
                            domain = determine_document_domain(fulltext)
                            st.session_state['domain'] = domain
                        st.info(f"Document Domain: **{domain}**")
                        with st.spinner("Searching for related Wikipedia articles..."):
                            articles_list = search_web_for_articles(domain)
                            st.session_state['articles_text'] = articles_list
                        if articles_list:
                            st.success("Found external Wikipedia articles:")
                            for article in articles_list:
                                with st.expander(article["title"]):
                                    st.markdown(f"[Link to article]({article['link']})")
                                    st.write(article["content"])
                        else:
                            st.warning("No external articles found.")
                    else:
                        articles_list = []
                        st.info("Using document text only for KB in AskMe Sidebar.")
                        st.session_state['articles_text'] = articles_list
                    # Build combined KB for AskMe Sidebar using document text + external articles if available
                    combined_text = fulltext
                    if st.session_state['articles_text']:
                        wiki_text = "\n".join([article["content"] for article in st.session_state['articles_text']])
                        combined_text = fulltext + "\n" + wiki_text
                    with st.spinner("Building combined knowledge base..."):
                        kb = create_knowledge_base(fulltext, st.session_state['articles_text'])
                        st.session_state['knowledge_base'] = kb
                    st.success("Knowledge base built successfully.")
                else:
                    st.error("Failed to extract text from the document.")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Document Search Results (Based on Combined KB)")
    
    if search_button:
        fulltext = st.session_state.get('fulltext', "")
        articles_list = st.session_state.get('articles_text', [])
        combined_text = fulltext
        if articles_list:
            wiki_text = "\n".join([article["content"] for article in articles_list])
            combined_text = fulltext + "\n" + wiki_text
        question_search = st.session_state.get('search_query')
        if question_search.strip() and combined_text:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            texts = text_splitter.split_text(combined_text)
            vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})
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
            st.markdown("*Answer based on Combined KB (Document + Wikipedia):*")
            with st.expander(f"Document Answer ({len(document_answer)} characters)"):
                st.markdown(document_answer)
            st.markdown("*Gemini Answer:*")
            with st.expander(f"Gemini Answer ({len(gemini_answer)} characters)"):
                st.markdown(gemini_answer)

# --------------------- Main Page Tabs: Generate Question & Quiz Mode ---------------------
# For Generate Question and Quiz, use only the document text.
doc_text_only = st.session_state.get('fulltext', "")

tab_generate, tab_quiz = st.tabs(["Generate Question", "Quiz Mode"])

# ----- Tab: Generate Question (Document Only) -----
with tab_generate:
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;">
            <h2 style="color: #333;">Generate a New Question (Document Only)</h2>
        </div>
        """, unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        Clicked = st.button("Generate New Question", key="new_question_doc", help="Click to generate a new question")
        if Clicked:
            if doc_text_only == "":
                st.warning("Please extract text first.")
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = text_splitter.split_text(doc_text_only)
                context_text = random.choice(chunks) if chunks else doc_text_only
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
    with col2:
        if st.session_state.generate_question:
            Clicked_Harder = st.button("Generate Harder Question", key="hard_question_doc", help="Click to generate a harder question")
            if Clicked_Harder:
                st.session_state.generated = False
                st.session_state.generated_hard = True
                st.session_state.generated_easy = False
                prev_q = st.session_state.get('question_output', "")
                prev_c = st.session_state.get('correct_answer', "")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = text_splitter.split_text(doc_text_only)
                context_text = random.choice(chunks) if chunks else doc_text_only
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
    with col3:
        if st.session_state.generate_question:
            Easier_Clicked = st.button("Generate Easier Question", key="easy_question_doc", help="Click to generate an easier question")
            if Easier_Clicked:
                st.session_state.generated = False
                st.session_state.generated_hard = False
                st.session_state.generated_easy = True
                prev_q = st.session_state.get('question_output', "")
                prev_c = st.session_state.get('correct_answer', "")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = text_splitter.split_text(doc_text_only)
                context_text = random.choice(chunks) if chunks else doc_text_only
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

# ----- Tab: Quiz Mode (Document Only) -----
with tab_quiz:
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;">
            <h2 style="color: #333;">Document-Based Quiz (Document Only)</h2>
            <p style="font-size: 16px; color: #555;">Answer 10 multiple-choice questions generated solely from your uploaded document.</p>
        </div>
        """, unsafe_allow_html=True
    )
    if st.button("Start Quiz / Restart Quiz", key="start_quiz_doc"):
        st.session_state.quiz_questions = []
        st.session_state.quiz_index = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_history = []
        if doc_text_only:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            chunks = text_splitter.split_text(doc_text_only)
            for i in range(10):
                context_text = random.choice(chunks) if chunks else doc_text_only
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
            if st.button("Restart Quiz", key="restart_quiz_doc"):
                st.session_state.quiz_questions = []
                st.session_state.quiz_index = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_history = []
                st.experimental_rerun()

st.write("---")
