import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM # For loading merged PEFT models

from pyngrok import ngrok
ngrok.set_auth_token("2ztThT7x7TDygNjIO4cUt920Woa_3gEdFpiSo1E3rFkXjidtv")

# --- Configuration ---
# Set your Hugging Face model ID (your_username/your_fine_tuned_model_repo)
# This should be the repo where you pushed your fine-tuned model (merged with base model).
HUGGINGFACE_MODEL_ID = "haranboyaiml/mistral-7b-recipes-quantized"

# --- Model Loading (Cached for efficiency) ---
# Use Streamlit's caching mechanism to load the model only once.
@st.cache_resource
def load_model():
    """
    Loads the merged fine-tuned model and tokenizer from Hugging Face with 4-bit quantization.
    This function is cached by Streamlit to run only once.
    """
    st.write("Loading model... This might take a few minutes.")

    # 4-bit quantization configuration (must match training setup)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, # Use float16 for T4 GPU compatibility
        bnb_4bit_use_double_quant=False,
        llm_int8_enable_fp32_cpu_offload=True, # Allows offloading some 32-bit modules to CPU if needed
    )

    # Load the merged model directly from Hugging Face Hub
    model = AutoPeftModelForCausalLM.from_pretrained(
        HUGGINGFACE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto" # Automatically distribute model layers across available devices
    )

    # Load the tokenizer associated with the model
    tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Important for generation tasks

    model.eval() # Set model to evaluation mode
    st.success("Model loaded successfully!")
    return model, tokenizer

# --- Text Generation Function ---
def generate_recipe(prompt_text, model, tokenizer, max_new_tokens=500, temperature=0.7, top_p=0.9):
    """
    Generates text (recipe) based on the input prompt using the loaded model.
    """
    if model is None or tokenizer is None:
        st.error("Model not loaded. Please wait for the model to load.")
        return "Error: Model not ready."

    # Format the prompt to match the Mistral instruction format
    formatted_prompt = f"[INST] {prompt_text} [/INST]"

    # Tokenize input and move to GPU
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process to remove the input prompt if it's included in the output
    if generated_text.startswith(formatted_prompt):
        generated_text = generated_text[len(formatted_prompt):].strip()

    return generated_text

# --- Streamlit UI ---
st.set_page_config(page_title="Mistral Recipe Generator", layout="centered")

st.title("🍜 Mistral Recipe Generator")
st.markdown("Enter a prompt below to generate a recipe!")

# Load model and tokenizer
model, tokenizer = load_model()

# Input for the recipe prompt
prompt = st.text_area(
    "Recipe Prompt (e.g., 'A quick and easy chicken curry recipe', 'How to make vegan chocolate chip cookies')",
    height=100,
    value="Generate a recipe for a simple breakfast smoothie with berries and banana."
)

# Generation parameters
st.sidebar.header("Generation Parameters")
max_tokens = st.sidebar.slider("Max New Tokens", 50, 1000, 500)
temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7)
top_p = st.sidebar.slider("Top P", 0.1, 1.0, 0.9)

# Generate button
if st.button("Generate Recipe"):
    if prompt:
        with st.spinner("Generating recipe..."):
            recipe = generate_recipe(prompt, model, tokenizer, max_tokens, temperature, top_p)
            st.subheader("Generated Recipe:")
            st.write(recipe)
    else:
        st.warning("Please enter a prompt to generate a recipe.")

st.markdown("---")
st.markdown("Powered by Mistral 7B and Streamlit")
