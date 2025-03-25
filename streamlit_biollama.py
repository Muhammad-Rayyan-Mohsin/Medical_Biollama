import streamlit as st
import transformers
import torch
import time
from datetime import datetime
import base64
import random

# --- Custom CSS and Theme Setup ---
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded_string});
        background-size: cover;
    }}
    </style>
    """

def local_css():
    st.markdown("""
    <style>
    .main {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    .stChatMessage {
        background-color: rgba(240, 242, 246, 0.9) !important;
        border-radius: 15px !important;
        padding: 15px !important;
        animation: fadeIn 0.5s;
        transition: all 0.3s ease;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stChatMessage:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .user-bubble .stChatMessage {
        background-color: rgba(232, 245, 253, 0.9) !important;
        border: 1px solid #e6f3fb !important;
    }
    .assistant-bubble .stChatMessage {
        background-color: rgba(232, 253, 240, 0.9) !important;
        border: 1px solid #e6fbe9 !important;
    }
    .stSpinner > div > div {
        border-top-color: #4285F4 !important;
    }
    .css-18e3th9 {
        padding-top: 2rem;
        padding-bottom: 10rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .css-1d391kg {
        padding-top: 3.5rem;
        padding-right: 1rem;
        padding-bottom: 3.5rem;
        padding-left: 1rem;
    }
    .chat-header {
        background: linear-gradient(90deg, #3a7bd5, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Arial', sans-serif;
        font-weight: 800;
        margin-bottom: 10px;
        padding: 10px;
        text-align: center;
    }
    .sidebar .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4285F4;
        color: white;
        font-weight: bold;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    .sidebar .stButton > button:hover {
        background-color: #3b77db;
        transform: scale(1.02);
    }
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    .typing-indicator span {
        height: 8px;
        width: 8px;
        margin: 0 2px;
        background-color: #3b77db;
        border-radius: 50%;
        display: inline-block;
        animation: bounce 1.5s infinite ease-in-out;
    }
    .typing-indicator span:nth-child(1) {
        animation-delay: 0s;
    }
    .typing-indicator span:nth-child(2) {
        animation-delay: 0.2s;
    }
    .typing-indicator span:nth-child(3) {
        animation-delay: 0.4s;
    }
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    .chat-container {
        margin-bottom: 100px;
        padding-bottom: 40px;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-top: 1px solid #e6e9ef;
        padding: 10px 0;
        z-index: 100;
    }
    .info-box {
        background-color: rgba(245, 245, 250, 0.9);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 5px solid #4285F4;
    }
    .model-info {
        font-size: 0.8em;
        color: #555;
        text-align: center;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Page Configuration ---
st.set_page_config(
    page_title="BioMedical AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to load custom background - wrapped in try/except in case file doesn't exist
try:
    # st.markdown(add_bg_from_local('medical_bg.png'), unsafe_allow_html=True)
    pass  # Comment this line and uncomment the above if you have a background image
except FileNotFoundError:
    pass  # No background image, will use default

local_css()

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown('<h2 class="chat-header">⚕️ BioMedical AI Settings</h2>', unsafe_allow_html=True)
    
    # Model parameters
    st.subheader("Generation Parameters")
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.6, step=0.1, 
                           help="Higher values make output more random, lower values more deterministic")
    max_tokens = st.slider("Max Response Length", min_value=64, max_value=512, value=256, step=32, 
                          help="Maximum number of tokens in the model's response")
    top_p = st.slider("Top P", min_value=0.1, max_value=1.0, value=0.9, step=0.1,
                     help="Controls diversity via nucleus sampling")
    
    # Chat controls
    st.subheader("Chat Controls")
    if st.button("🔄 Reset Conversation", key="reset_sidebar"):
        st.session_state.messages = [
            {"role": "system", "content": "You are an expert trained on healthcare and biomedical domain!"}
        ]
        st.session_state.chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
        st.rerun()
    
    # Theme toggle
    st.subheader("Interface")
    theme = st.selectbox("Theme", ["Light", "Dark"], 
                       help="Choose interface theme (may require page reload)")
    if theme == "Dark":
        st.markdown("""
        <style>
        .main {background-color: rgba(25, 25, 25, 0.9);}
        .stChatMessage {background-color: rgba(45, 45, 45, 0.9) !important; color: #fff !important;}
        .user-bubble .stChatMessage {background-color: rgba(55, 65, 81, 0.9) !important; border: 1px solid #4b5563 !important;}
        .assistant-bubble .stChatMessage {background-color: rgba(31, 41, 55, 0.9) !important; border: 1px solid #374151 !important;}
        .info-box {background-color: rgba(31, 41, 55, 0.9); color: #e0e0e0;}
        </style>
        """, unsafe_allow_html=True)
    
    # About section
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <h4>About</h4>
    <p>This AI assistant uses Bio-Medical Llama 3, a specialized language model fine-tuned for healthcare and biomedical conversations.</p>
    <p>ℹ️ This is not a substitute for professional medical advice.</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are an expert trained on healthcare and biomedical domain!"}
    ]

if "chat_id" not in st.session_state:
    st.session_state.chat_id = datetime.now().strftime("%Y%m%d%H%M%S")

# --- Main UI ---
st.markdown('<h1 class="chat-header">🩺 BioMedical AI Assistant</h1>', unsafe_allow_html=True)

# Greeting message at the start of a new conversation
if len(st.session_state.messages) == 1:  # Only system message exists
    st.markdown("""
    <div class="info-box">
    <p>👋 Welcome! I'm a BioMedical AI Assistant trained to help with healthcare and biomedical questions. 
    Ask me about medical conditions, treatments, or general health information.</p>
    <p><strong>Note:</strong> I'm here to provide information, but not to replace professional medical advice.</p>
    </div>
    """, unsafe_allow_html=True)

# Display chat messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    if message["role"] != "system":  # Don't display system messages
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🩺"):
            st.markdown(message["content"])
st.markdown('</div>', unsafe_allow_html=True)

# --- Initialize model with caching ---
@st.cache_resource
def initialize_model():
    model_id = "ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025"
    try:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        return pipeline, None
    except Exception as e:
        return None, str(e)

# --- Custom typing indicator ---
def display_typing_animation():
    typing_container = st.empty()
    typing_container.markdown("""
    <div class="typing-indicator">
        <span></span>
        <span></span>
        <span></span>
    </div>
    """, unsafe_allow_html=True)
    return typing_container

# --- Footer with input ---
st.markdown('<div class="footer">', unsafe_allow_html=True)
# User input
user_question = st.chat_input("Enter your healthcare or medical question...")
st.markdown('</div>', unsafe_allow_html=True)

# Handle user input and generate response
if user_question:
    # Add user message to chat and session state
    st.chat_message("user", avatar="👤").markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    # Show typing indicator
    typing_indicator = display_typing_animation()
    
    # Initialize model pipeline
    pipeline, error = initialize_model()
    
    if error:
        typing_indicator.error(f"Error loading model: {error}")
    else:
        try:
            # Generate response with user-configured parameters
            prompt = pipeline.tokenizer.apply_chat_template(
                st.session_state.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            # Use the parameters from the sidebar
            outputs = pipeline(
                prompt,
                max_new_tokens=max_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            
            # Extract the response
            response = outputs[0]["generated_text"][len(prompt):]
            
            # Add a small delay for natural typing effect
            time.sleep(0.5)
            typing_indicator.empty()
            
            # Display assistant response
            st.chat_message("assistant", avatar="🩺").markdown(response)
            
            # Add assistant response to session state
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            typing_indicator.empty()
            st.error(f"An error occurred: {str(e)}")

# Display model information at the bottom
st.markdown("""
<p class="model-info">Powered by Bio-Medical Llama 3 | Model: ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025</p>
""", unsafe_allow_html=True)
