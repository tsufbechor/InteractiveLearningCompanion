
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

- **Document Querying with Retrieval-Augmented Generation (RAG):**  
  Retrieve relevant information from your uploaded document and get a separate Gemini-based answer for broader context.
  The AskMe Sidebar uses a knowledge base built from your document.
-**Optional KB Enrichment with Wikipedia:**
  By default, the knowledge base (KB) is built solely from your document. Users can opt in via a checkbox to enrich the AskMe Sidebar KB with up to 10 related Wikipedia articles. When enabled, the app determines the document’s domain, retrieves relevant Wikipedia articles, and rebuilds the KB using the combined content. External articles are displayed in separate expandable panels for easy exploration.
- **Integration with Gemini:**  
  Utilizes Gemini for text generation (Gemini Pro) and Gemini Pro Vision for image-based text extraction.
- **Chain of Thought Explanation for Answers:**
  The model provides to the user a "Chain of Thought" that explains the steps the model took to decide the correct answer
## Technologies Used

- **LangChain Framework:**  
  For seamless integration of Large Language Models (LLMs) and advanced prompting.
  
- **RAG (Retrieval-Augmented Generation):**  
  Enabling real-time access to updated and reliable content from your document (and optionally from Wikipedia).

- **Gemini Models:**  
  Tailored for high-quality question-answer generation (Gemini Pro) and image-based text extraction (Gemini Pro Vision).

- **Streamlit:**  
  Providing an intuitive user interface, tabbed layout, and real-time feedback.

## Use Cases

- **Students**  
  - Generate custom quizzes to practice newly learned material.  
  - Explore “harder” or “easier” questions to adapt to your current understanding.
  - Use document querying to ask question and get answer based on document/knowledge base with similar documents/or gemini model

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


## Examples of Tool in Use:
1. **User Answers Question:** 
![image](https://github.com/user-attachments/assets/0920530b-d2fe-42ee-b45a-3aa9f1f6014d)
2. **Model Provides Chain Of thought in explanation**
![image](https://github.com/user-attachments/assets/e9324879-7852-4aa7-87f7-388c4142485c)
3. **User Requests Harder Question:**
   ![image](https://github.com/user-attachments/assets/06020f55-6f1c-444b-9e65-eed3f5dd8206)
4. **Quiz Mode:**
   ![image](https://github.com/user-attachments/assets/ce6b72b3-1e03-41d2-a3b3-e9d3be653caa)
5. **Building Knowledge Base automatically with Wikipedia API**
   ![image](https://github.com/user-attachments/assets/05c14161-fe84-4b6e-ae19-3764fefa22e9)
6. **User Querying, 1 answer based on Knowledge Base and 1 based on Gemini**
   ![image](https://github.com/user-attachments/assets/bfed2371-ef02-4a7d-a5a5-0e02da04906b)
   ![image](https://github.com/user-attachments/assets/76406456-8aeb-4c2e-ab9a-c2fafe54fce9)
