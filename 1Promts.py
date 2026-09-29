from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt


# Load .env file
load_dotenv()


# Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=1000
)


# Create Chat Model
model = ChatHuggingFace(llm=llm)


# Load prompt template
tem = load_prompt("1.LLMs1/template.json")


# Streamlit heading
st.header("Research Paper")


# Select research paper
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)


# Select explanation style
style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)


# Select explanation length
length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)


# Submit button
if st.button("Submit"):

    st.write("⏳ Generating answer...")

    # Create chain
    chain = tem | model

    # Send input to model
    result = chain.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input
    })

    # Display result
    st.success("Answer generated!")

    st.write(result.content)