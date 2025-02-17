
# Interactive Learning Companion

A virtual learning companion to transform passive learning into active engagement. Designed for students, teachers, and curious learners, this app personalizes the educational experience by dynamically generating questions and providing document-based answers tailored to user preferences and interactions.

## Features

- **Document and Image Upload:** Upload PDF documents or text images for processing.
- **Dynamic Question Generation:** Create questions with adjustable difficulty levels for enhanced learning.
- **Personalized Learning:** Adaptive feedback and question difficulty based on user interactions.
- **Document Querying:** Retrieve relevant information from uploaded documents or use RAG for external context.
- **Integration with Gemini:** Utilizes Gemini for text generation and Gemini Pro Vision for image-based text extraction.

## Technologies Used

- **LangChain Framework:** For seamless integration of Large Language Models (LLMs).
- **RAG (Retrieval-Augmented Generation):** Enabling real-time access to updated and reliable content.
- **Gemini Models:** Tailored for high-quality question-answer generation.
- **Streamlit:** Providing an intuitive user interface.
## Use Cases
- **Students**: Generate custom quizzes and review key topics interactively.
- **Teachers**: Create tailored exercises and track student performance.
- **Independent Learners**: Explore new subjects with adaptive questioning and immediate feedback.
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
