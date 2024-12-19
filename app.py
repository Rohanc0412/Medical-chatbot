import streamlit as st # type: ignore
from rag import Rag




def display_messages():
    for messages in st.session_state.messages:
        with st.chat_message(messages["role"]):
            st.markdown(messages["content"])


def process_input():
    if prompt := st.chat_input("How can i help?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})


        response = st.session_state["assistant"].ask(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)


        st.session_state.messages.append({"role": "assistant", "content": response})




def main():
    st.title("Medical Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize the assistant (Rag) if not already initialized
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = Rag()  # Initialize the Rag system
    
    display_messages()
    process_input()





if __name__ == "__main__":
    main()