import os
import streamlit as st
import fitz  # PyMuPDF
import PyPDF2
import google.generativeai as genai
import PIL.Image
from langchain.chains.llm import LLMChain
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types import StopCandidateException
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationChain
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter


st.set_page_config(page_title="AskMe", layout="wide")
# st.title("📘 Lets Study!")
TEMP_DIR = "temp"
QUESTION_LEN = 9
CORRECT_LEN = 14
ANSWER_LEN = 7

# Define a session_state variable to store the extracted text
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
if 'answered_question' not in st.session_state:
    st.session_state.answered_question = False
if 'question_output' not in st.session_state:
    st.session_state.question_output = "What is the capital of France?"
if 'answers' not in st.session_state:
    st.session_state.answers = "A) Paris", "B) Madrid", "C) Berlin", "D) Rome"
if 'correct_answer' not in st.session_state:
    st.session_state.correct_answer = "A) Paris"


def handle_answer_selection(selected_answer):
    # Mark the question as answered
    st.session_state['answered'] = True
    # Store the selected answer
    st.session_state['selected_answer'] = selected_answer
    # Determine if the selected answer is correct
    correct_answer = st.session_state['correct_answer']
    if selected_answer in correct_answer:
        st.session_state['is_correct'] = True
    else:
        st.session_state['is_correct'] = False


if 'answered' not in st.session_state:
    st.session_state['answered'] = False
if 'selected_answer' not in st.session_state:
    st.session_state['selected_answer'] = ''
if 'is_correct' not in st.session_state:
    st.session_state['is_correct'] = False


def change_upload_img_state():
    if st.session_state['upload_img'] == 'not done':
        st.session_state['upload_img'] = 'done'


def reformat_output(output):
    # this function will separate the output to question,answers,correct format
    output = output.replace("*", "")
    index_after_q_mark = output.find('?') + 1
    index_before_correct = output.find('Correct')
    index_before_answer = output.find('Answer')

    if output.find('Question:') == -1:
        question_txt = output[:index_after_q_mark]
    else:
        question_txt = output[QUESTION_LEN:index_after_q_mark]
    if index_before_correct == -1:
        answers_txt = output[index_after_q_mark:index_before_answer]
        correct_txt = output[index_before_answer + ANSWER_LEN:]
    else:
        answers_txt = output[index_after_q_mark:index_before_correct]
        correct_txt = output[index_before_correct + CORRECT_LEN:]
    return question_txt, answers_txt, correct_txt


def reformat_answers(answers_txt):
    if isinstance(answers_txt, list):
        return answers_txt[0], answers_txt[1], answers_txt[2], answers_txt[3]
    else:
        index_a = answers_txt.find('(A)')
        index_b = answers_txt.find('(B)')
        index_c = answers_txt.find('(C)')
        index_d = answers_txt.find('(D)')

        ans_a = answers_txt[index_a:index_b].replace("\n", "")
        ans_b = answers_txt[index_b:index_c].replace("\n", "")
        ans_c = answers_txt[index_c:index_d].replace("\n", "")
        ans_d = answers_txt[index_d:].replace("\n", "")
        return ans_a, ans_b, ans_c, ans_d


def question_anwered():
    st.session_state.answered_question = True
    st.write("YOU ANSWERED!")


def display_question_answer(question, answers, correct_answer):
    # Custom CSS for styling
    st.session_state['answers'] = ""
    ans_a, ans_b, ans_c, ans_d = reformat_answers(answers)
    st.session_state['answers'] = [ans_a, ans_b, ans_c, ans_d]
    correct_let = ''

    st.session_state['correct_answer'] = correct_answer
    st.markdown("""
        <style>
            .question-container, .answer-container, .correct-answer-container {
                font-family: Arial, Helvetica, sans-serif;
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0px;
                border-left: 5px solid #0078D4;
            }
            .question-container {
                background-color: #f3f4f6;
            }
            .answer-container {
                background-color: #e2e3e5;
            }
            .correct-answer-container {
                background-color: #d1ecf1;
                border-left-color: #0c5460;
            }
            .header {
                font-weight: 600;
                font-size: 18px;
                margin-bottom: 10px;
            }
            .content {
                font-size: 16px;
            }
        </style>
        """, unsafe_allow_html=True)
    base_key = str(abs(hash(question)))[:10]
    # Displaying the styled question
    st.markdown(
        f"<div class='question-container'><div class='header'>Question</div><div class='content'>{question}</div></div>",
        unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    answer_buttons = [st.button(answer, key=f'answer_{idx}_{base_key}') for idx, answer in
                      enumerate(st.session_state.answers)]
    # answer_buttons = [st.button(answer, key=f'answer_{idx}_{uuid.uuid4()}') for idx, answer in
    #                   enumerate(st.session_state.answers)]

    # Check which button was pressed
    for idx, was_pressed in enumerate(answer_buttons):
        if was_pressed:
            handle_answer_selection(st.session_state.answers[idx])
            break  # Exit the loop if a button was pressed

    # Display feedback after an answer is selected
    if st.session_state['answered']:
        if st.session_state['is_correct']:
            st.success("Correct!")
        else:
            st.error(f"Incorrect. The correct answer is: {st.session_state['correct_answer']}")
        st.session_state['answered'] = False


def extract_text_from_pdf(doc_path, start_page, end_page):
    text = ""
    with open(doc_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(start_page - 1, end_page):
            if page_num < len(pdf_reader.pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    return text


def extract_text_from_photo(image_path, vision_model):
    try:
        img_pil = PIL.Image.open(image_path)
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Save the text shown in the image"
                },
                {"type": "image_url", "image_url": img_pil}
            ]
        )
        output = vision_model.invoke([message])
        content = output.content
        return content
    except StopCandidateException:
        return "error"


st.write("---")

st.header('Expand your knowledge')
st.markdown(f"**Workflow:** ")
st.markdown('1. Go to sidebar and choose desired document or photo\n'
            '2. Extract the text\n'
            '3. Use one of the following:\n'
            '- Search for an answer to a specific question using the sidebar Document Search\n'
            '- Get an AI generated question using the Generate New Question button in the middle of the page ')
st.write("---")
# uploaded_files = os.listdir(TEMP_DIR)
# selected_file = st.selectbox("Select a file", uploaded_files)
with st.sidebar:
    uploaded_files = os.listdir(TEMP_DIR)
    selected_file = st.selectbox("Select a file to analyze:", uploaded_files)

API_KEY = 'Enter API KEY here'
genai.configure(api_key = API_KEY)


model = ChatGoogleGenerativeAI(model="gemini-pro",
                               google_api_key=API_KEY, convert_system_message_to_human=True,
                               temperature=0.4)
vision_model = ChatGoogleGenerativeAI(model="gemini-pro-vision", google_api_key=API_KEY)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True, google_api_key=API_KEY,
                             temperature=0.4)
memory = ConversationEntityMemory(llm=llm)
conversation = ConversationChain(llm=llm, prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE, memory=memory)
question = None

if selected_file:
    file_path = os.path.join(TEMP_DIR, selected_file)

    with st.sidebar:
        if selected_file.lower().endswith(".pdf"):
            doc_path = os.path.join(TEMP_DIR, selected_file)
            pdf_document = fitz.open(file_path)

            st.write("We recommend choosing a small amount of pages for fast and convenient loading.")
            col1, col2 = st.columns(2)
            with col1:
                start_page = st.number_input("Enter start page", value=1, min_value=1,
                                             max_value=pdf_document.page_count,
                                             step=1)
            with col2:
                end_page = st.number_input("Enter end page", value=pdf_document.page_count, min_value=1,
                                           max_value=pdf_document.page_count, step=1)
        else:
            image_path = os.path.join(TEMP_DIR, selected_file)

        if st.button("Extract Text"):
            st.session_state['display_text'] = True
            if selected_file.lower().endswith(".pdf"):
                fulltext = extract_text_from_pdf(doc_path, start_page, end_page)
                st.session_state['fulltext'] = fulltext
                if start_page <= end_page:
                    # pdf_document.close()
                     with st.expander(f"View extracted text ({len(fulltext)} characters)"):
                        st.write(fulltext)
                else:
                    st.warning("End page must be greater than or equal to start page.")
            else:
                fulltext = extract_text_from_photo(file_path, vision_model)
                if fulltext == "error":
                    st.session_state['fulltext'] = ""
                    st.error('Failed to extract text from the image. Please try again with a different image.',
                             icon="🚨")
                else:
                    st.session_state['fulltext'] = fulltext
                    with st.expander(f"View extracted text ({len(fulltext)} characters)"):
                        st.write(fulltext)
        else:
            if st.session_state.get('display_text', False):
                fulltext = st.session_state.get('fulltext', "")
                if selected_file.lower().endswith(".pdf") and start_page > end_page:
                    st.warning("End page must be greater than or equal to start page.")
                elif fulltext:  # Check if fulltext is not empty
                    # with st.expander(f"View extracted text ({len(fulltext)} characters)"):
                    with st.expander(f"View extracted text ({len(fulltext)} characters)"):
                        st.write(fulltext)

    with st.sidebar:
        st.title("Document Search")
        question = st.text_input("Enter your question here:", "", on_change=None)
        search_button = st.button("Search")
        if search_button:
            question = st.session_state.get('question')
            if not question.strip():
                st.warning("Please enter a non-empty question.")
            elif st.session_state['fulltext'] == "":
                st.warning("Please extract text first.")

    if question:
        st.session_state['question'] = question

    if search_button:
        fulltext = st.session_state.get('fulltext')
        question = st.session_state.get('question')
        st.session_state['question'] = ""
        if not question.strip():
            pass
        elif st.session_state['fulltext'] == "":
            pass
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            texts = text_splitter.split_text(fulltext)
            vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=vector_index,
                return_source_documents=True
            )
            result = qa_chain({"query": question})
            question_result = result["result"]
            template = """Please answer the following question:
                         {question}
                        """
            prompt = ChatPromptTemplate.from_template(template)
            chain = LLMChain(llm=llm, prompt=prompt)
            output = chain.invoke(question)['text']
            st.session_state['fulltext'] += output
            if question_result:
                st.markdown(f"**Answer based on Document:** ")
                with st.expander(f"View document extracted answer ({len(question_result)} characters)"):
                    st.markdown(question_result)
                st.markdown(f"**Gemini Answer:**")
                with st.expander(f"View gemini extracted answer ({len(output)} characters)"):
                    st.markdown(output)

    st.write('---')
    # st.header("🤔 Generate Question")
    st.markdown("""
    <div style="text-align: center;">
        <h1>🤔 Lets see how smart you are</h1>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        Clicked = st.button("Generate New Question", key="new_question", help="Click to generate a new question")
        if Clicked:
            if st.session_state['fulltext'] == "":
                st.warning("Please extract text first.")
            else:
                template = """Generate an easy multiple choice question, with 4 possible answers, based on the following text:
                {text}
                "Also  then write the correct answer
                """
                texts = st.session_state.get('fulltext')
                prompt = ChatPromptTemplate.from_template(template)
                chain = LLMChain(llm=llm, prompt=prompt)
                text1 = texts
                output = chain.invoke(text1)['text']
                # st.text_area("Review the generated question and answer below:", value=output,
                #              height=200)
                # st.success("Question generated successfully!")
                question, answers, correct_answer = reformat_output(output)
                # reformat_answers(answers)
                st.session_state['question_output'], st.session_state['answers'], st.session_state[
                    'correct_answer'] = question, answers, correct_answer
                st.session_state.generate_question = True
                st.session_state.generated = True
        if st.session_state.generated:
            st.session_state.generated_easy = False
            st.session_state.generated_hard = False
            question = st.session_state['question_output']
            answers = st.session_state['answers']
            correct_answer = st.session_state['correct_answer']
            display_question_answer(question, answers, correct_answer)

    with col2:
        if st.session_state.generate_question:
            Clicked_Harder = st.button("Generate Harder Question", key="hard_question",
                                       help="Click to generate a harder question")
            if Clicked_Harder:
                st.session_state.generated = False
                st.session_state.generated_easy = False
                question_output = st.session_state.get('question_output')
                correct_answer = st.session_state.get('correct_answer')
                context_text = st.session_state.get('fulltext')
                output = conversation.predict(
                    input=f"""
                    This is the previous question:{question_output}. The user answered correctly with the answer:{correct_answer}. Generate  different, more difficult,multiple choice question with 4 possible answers
                    about subject. provide correct answer
                    """
                )
                # st.write(predicted)
                question, answers, correct_answer = reformat_output(output)
                st.session_state['question_output'], st.session_state['answers'], st.session_state[
                    'correct_answer'] = question, answers, correct_answer
                st.session_state.generated_hard = True
        if st.session_state.generated_hard:
            st.session_state.generate_question = False
            st.session_state.generated = False
            st.session_state.generated_easy = False
            question = st.session_state['question_output']
            answers = st.session_state['answers']
            correct_answer = st.session_state['correct_answer']
            display_question_answer(question, answers, correct_answer)

    with col3:
        if st.session_state.generate_question:
            Easier_Clicked = st.button("Generate Easier Question", key="easy_question",
                                       help="Click to generate an easier question")
            if Easier_Clicked:
                st.session_state.generated = False
                st.session_state.generated_hard = False
                question_output = st.session_state.get('question_output')
                correct_answer = st.session_state.get('correct_answer')
                context_text = st.session_state.get('fulltext')
                output = conversation.predict(
                    input=f"""
                    This is the previous question:{question_output}.The user answered incorrectly with: {correct_answer}.Generate different,easier ,multiple choice question with 4 possible answers
                    about subject. provide correct answer
                    """
                )
                question, answers, correct_answer = reformat_output(output)
                st.session_state['question_output'], st.session_state['answers'], st.session_state[
                    'correct_answer'] = question, answers, correct_answer
                st.session_state.generated_easy = True

        if st.session_state.generated_easy:
            st.session_state.generate_question = False
            st.session_state.generated = False
            st.session_state.generated_hard = False
            question = st.session_state['question_output']
            answers = st.session_state['answers']
            correct_answer = st.session_state['correct_answer']
            display_question_answer(question, answers, correct_answer)

st.write("---")
