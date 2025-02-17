
# Interactive Learning Companion

A virtual learning companion to transform passive learning into active engagement. Designed for students, teachers, and curious learners, this app personalizes the educational experience by dynamically generating questions and providing document-based answers tailored to user preferences and interactions.

## Features

- **Document and Image Upload:**  
  Upload PDF documents or text images for processing. The app can extract text page-by-page (for PDFs) or through Gemini Pro Vision (for images).

- **Tabbed Interface:**  
  The main page is split into two primary tabs:  
  1. **Generate Question** – Create a single question with adjustable difficulty levels (New/Harder/Easier).  
  2. **Quiz Mode** – Generate a 10-question quiz from your uploaded document, track progress with a progress bar, and view a final performance summary.

- **Dynamic Question Generation:**  
  - **Generate New Question**: Creates a multiple-choice question from a random context chunk in your uploaded text.  
  - **Generate Harder Question**: Produces a more challenging question than the previous one.  
  - **Generate Easier Question**: Produces a simpler question than the previous one.  

- **Personalized Learning:**  
  Adaptive feedback and question difficulty based on user interactions. Immediate correct/incorrect notifications with each question.

- **Document Querying:**  
  Retrieve relevant information from uploaded documents (via Retrieval-Augmented Generation) and get a separate Gemini-based answer for broader context.

- **Integration with Gemini:**  
  Utilizes Gemini for text generation (Gemini Pro) and Gemini Pro Vision for image-based text extraction.

## Technologies Used

- **LangChain Framework:**  
  For seamless integration of Large Language Models (LLMs) and advanced prompting.
  
- **RAG (Retrieval-Augmented Generation):**  
  Enabling real-time access to updated and reliable content from your document.

- **Gemini Models:**  
  Tailored for high-quality question-answer generation (Gemini Pro) and image-based text extraction (Gemini Pro Vision).

- **Streamlit:**  
  Providing an intuitive user interface, tabbed layout, and real-time feedback.

## Use Cases

- **Students**  
  - Generate custom quizzes to practice newly learned material.  
  - Explore “harder” or “easier” questions to adapt to your current understanding.

- **Teachers**  
  - Create tailored exercises from a classroom PDF or image-based handout.  
  - Quickly produce quiz questions of varying difficulty and track student performance.

- **Independent Learners**  
  - Engage in a self-paced, adaptive question flow.  
  - Use the quiz mode to test comprehension and monitor progress.
## Getting Started

1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/tsufbechor/InteractiveLearningCompanion.git
   cd InteractiveLearningCompanion

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   Enter you API KEY in generate.py
3. **Run the Application:**
   ```bash
   streamlit run streamlit_app.py --server.enableXsrfProtection false


## Future Enhancements
Enhanced memory for tracking long-term user interactions.
Insights page for reviewing past performance and question analytics.
Integration of specialized knowledge bases for niche academic topics.
