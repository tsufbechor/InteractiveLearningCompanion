import os
import streamlit as st
import time
import fitz  # PyMuPDF
import os


TEMP_DIR = "temp"

def change_upload_pdf_state():
    if st.session_state['upload_pdf'] == 'not done':
        st.session_state['upload_pdf'] = 'done'



def extract_text_from_pdf(doc_path):
    text = ""
    pdf_document = fitz.open(doc_path)
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    return text


def is_image(file):
    # Check if the file is a photo based on the file extension
    allowed_extensions = ["jpg", "jpeg", "png", "gif"]
    file_extension = file.name.split(".")[-1].lower()
    if file_extension in allowed_extensions:
        return True

    # Check if the file is a photo based on the MIME type
    allowed_mime_types = ["image/jpeg", "image/png", "image/gif"]
    if file.type in allowed_mime_types:
        return True

    return False


def change_upload_img_state():
    if st.session_state['upload_img'] == 'not done':
        st.session_state['upload_img'] = 'done'

if "upload_pdf" not in st.session_state:
    st.session_state['upload_pdf'] = 'not done'

st.title("Upload Files")
st.write("---")

#PDF
st.markdown("<h3 style='text-align: left;'>PDF section</h3>", unsafe_allow_html=True)
doc = st.file_uploader("Upload a PDF document", type="pdf", on_change=change_upload_pdf_state)


if 'upload_pdf' in st.session_state and st.session_state['upload_pdf'] == 'done' and doc is not None:
    if os.path.splitext(doc.name)[1].lower() == ".pdf":
        # Create temporary directory if it does not exist
        os.makedirs(TEMP_DIR, exist_ok=True)
        # Save the uploaded file to the temporary directory
        doc_path = os.path.join(TEMP_DIR, doc.name)
        with open(doc_path, "wb") as f:
            f.write(doc.read())
        # Extract text from the PDF
        progress_bar = st.progress(0)
        for perc_comp in range(100):
            time.sleep(0.015)
            progress_bar.progress(perc_comp+1)
        st.success("Uploaded successfully!")
        st.write("You can now move to generate page.")
    else:
        st.warning("Please upload a PDF file.")


# PHOTO
st.write("---")

if "upload_img" not in st.session_state:
    st.session_state['upload_img'] = 'not done'

st.markdown("<h3 style='text-align: left;'>Photo section</h3>", unsafe_allow_html=True)
img = st.file_uploader("Upload a photo of PDF", type=["jpg", "jpeg", "png", "gif"],
                       on_change=change_upload_img_state)

if 'upload_img' in st.session_state and st.session_state['upload_img'] == 'done' and img is not None:
    if is_image(img):
        st.image(img, caption="Uploaded photo")
        # Create temporary directory if it does not exist
        os.makedirs(TEMP_DIR, exist_ok=True)

        # Save the uploaded file to the temporary directory
        image_path = os.path.join(TEMP_DIR, img.name)
        with open(image_path, "wb") as f:
            f.write(img.read())
        st.success("Uploaded successfully!")
        st.write("You can now move to generate page.")
    else:
        st.warning("Please upload a valid photo file (jpg, jpeg, png, gif).")

st.write("---")





