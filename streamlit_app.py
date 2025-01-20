import streamlit as st

def main():
    # Set page configuration
    st.set_page_config(page_title="AskMe", layout="wide")

    # Use columns to center content
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        # Custom styling
        st.markdown("""
            <style>
                .banner {
                    text-align: center;
                    margin: 25px 0;
                }
                .banner h1, .banner h3 {
                    margin: 10px 0;
                }
                .feature-icon {
                    font-size: 50px;
                    text-align: center;
                    margin: 20px 0;
                }
                .contact-info {
                    text-align: center;
                    margin-top: 20px;
                }
            </style>
        """, unsafe_allow_html=True)

        # Banner
        st.image("AskmeLogoHD.png", width=350)
        st.markdown("<div class='banner'><h1>Welcome to AskMe</h1><h3>Your AI companion for studying</h3></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='banner'>
                This app generates questions from uploaded PDFs and allows users to provide feedback
                to improve the questions. You can customize it with your own content, such as text and images.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("---")

        # About
        st.header("📘 About")
        st.write(
            """
            AskMe is an AI-powered app designed to help you study more effectively. By uploading
            your PDF documents or photos of them, AskMe can generate questions based on the content, helping you
            test your understanding and retention of the material.
            """
        )

        # Get Started
        st.header("🚀 Get Started")
        st.write(
            """
            To get started, simply upload a PDF document using the file uploader on the left. Once
            the document is uploaded, go to generate and AskMe will present the text and generate questions for you to answer.
            """
        )

        # Features
        st.header("✨ Features")
        st.write(
            """
            - Upload PDF documents or a photo of a PDF.
            - Pick the file or the relevant scope in it.
            - Generate questions.
            - Answer questions and receive instant feedback.
            - Provide feedback on questions to help improve future question generation.
            """
        )

        # Contact
        st.header("📞 Contact")
        st.markdown(
            """
            <div class='contact-info'>
                Have questions or feedback? Reach out to us at <a href='mailto:askme@gmail.com'>askme@gmail.com</a>
                or visit our website for more information.
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
