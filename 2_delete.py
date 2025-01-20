import os
import streamlit as st

TEMP_DIR = "temp"

def delete_file(file_name):
    file_path = os.path.join(TEMP_DIR, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)

st.title("Delete")
st.write("---")

uploaded_files = os.listdir(TEMP_DIR)
selected_file_name = st.selectbox("Select a file to delete", uploaded_files)
st.error('Attention! After deletion there is no way to recover the file through the website  ', icon="🚨")
if st.button("Delete"):
    delete_file(selected_file_name)
    st.success(f"Deleted {selected_file_name} successfully!")
    # Update the list of uploaded files after deleting the file
    uploaded_files = os.listdir(TEMP_DIR)
    # Check if there are still files left
    if uploaded_files:
        # Set the selected file to the first file in the list
        selected_file_name = uploaded_files[0]
    else:
        selected_file_name = None

st.write("---")
