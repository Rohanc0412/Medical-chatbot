from src.helper import load_pdf, text_split, download_embeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.schema.runnable import RunnablePassthrough
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os


load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

index_name = "medicalbot"
extracted_data = load_pdf(data='data/')
text_chunks = text_split(extracted_data)
embeddings = download_embeddings()



class NumberedWithSubPointsStrOutputParser(StrOutputParser):
    def parse(self, output: str) -> str:
        # Split the output string by newlines to handle each line individually
        lines = output.split('\n')
        
        # Initialize lists to keep track of the main points and sub-points
        formatted_lines = []
        current_main_point = None
        
        for line in lines:
            line = line.strip()  # Remove leading/trailing whitespaces
            
            if not line:  # Skip empty lines
                continue
            
            # Check if the line starts with a main point (e.g., "1.", "2.", etc.)
            if line[0].isdigit() and line[1] == '.':
                if current_main_point:
                    # If we already have a main point, add it to the formatted lines
                    formatted_lines.append(current_main_point)
                # Start a new main point
                current_main_point = f"{line}"
            # Check if the line starts with a sub-point (e.g., "a.", "b.", etc.)
            elif line[0].isalpha() and line[1] == '.':
                if current_main_point:
                    # Add sub-points under the current main point
                    formatted_lines.append(f"  {line}")  # Indent the sub-point
            else:
                # For other lines, just append to the last main point (if any)
                if current_main_point:
                    formatted_lines.append(f"  {line}")  # This is considered as additional content
            
        # Add the last main point (if exists)
        if current_main_point:
            formatted_lines.append(current_main_point)
        
        # Join all formatted lines into a single string and return it
        return '\n'.join(formatted_lines)



class Rag:
    retriever = None
    chain = None


    def __init__(self) -> None :
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. 
            Use the following piece of retrieved context to answer the question. If no relevant context is found, then say "I don't know" concisely. 
            The answer must only be based on the context provided. 
            Please list the points clearly and separate them by numbers.
            Keep the answer concise and answer in maximum 5 sentences.
            Strictly format the output as points.
            [/INST]</s>
            [INST] Question: {question}
            Context: {context}
            Answer (please list points in a properly numbered list): [/INST]
            """
        )

        self.model = ChatOllama(model="mistral")
        self.set_retriever()

    def set_retriever(self):
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding= embeddings
        )
        self.retriever = docsearch.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":3, "score_threshold": 0.7,})
        print("Retriever initialized: ", self.retriever)

    
    def test_retriever(self, query):
        """Test the retriever separately to see if it's fetching the correct documents."""
        results = self.retriever.g(query)
        print("Retriever results: ", results)
        
    
    def ask(self, question: str):
        """Get an answer from the model using dynamic context retrieved based on the question."""
        if self.retriever is None:
            raise AttributeError("Retriever has not been set. Please call set_retriever() first.")
        
        # Fetch context using the retriever based on the dynamic question
        test_results = self.retriever.get_relevant_documents(question)  # Using the dynamic question to fetch context
        print("Test results:", test_results)

        if test_results:
            # Concatenate the page content from multiple documents to create richer context
            context = "\n".join([doc.page_content for doc in test_results])  # Join content from all docs
        else:
            context = "I don't know."
            return "I don't know"
        
        # Ensure the context is passed properly
        print("Context for prompt: ", context)

        formatted_prompt = self.prompt.format(context=context, question=question)

        # Wrap the formatted prompt in a Runnable (no parameters, handles everything inside)
        prompt_runnable = RunnableLambda(lambda _: formatted_prompt)

        # Create the chain (prompt -> model -> output parser)
        self.chain = (
            prompt_runnable  # This is now a Runnable that generates the formatted prompt
            | self.model  # Pass it through the model for response generation
            | NumberedWithSubPointsStrOutputParser()  # Parse the output into the required format
        )

        print("Chain initialized with context:", context)

        # Running the chain with the provided question and context
        try:
            # Run the chain with the dynamic question and context
            response = self.chain.invoke({})
            print("Model response: ", response)
            return response
        except Exception as e:
            print(f"Error in chain invocation: {e}")
            return "I don't know"