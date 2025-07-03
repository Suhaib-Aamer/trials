# To run: streamlit run c:\Users\hi\Desktop\Streamlit_App\streamlit_app.py

from gensim.models.keyedvectors import KeyedVectors
import streamlit as st

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="Word2Vec Similarity Checker", layout="wide")

@st.cache_resource
def load_model(model_path):
    try:
        # Load the Word2Vec model once
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Path to your models
google_model_path = r"C:\Users\hi\Desktop\INNOMATICS\Machine Learning\Word2Vec\GoogleNews-vectors-negative300.bin"
custom_model_path = r"C:\Users\hi\Desktop\INNOMATICS\Machine Learning\Word2Vec\custom_w2v_model.bin"

# Model options for the sidebar
model_options = {
    "Google News Word2Vec": google_model_path,
    "My Custom Word2Vec Model": custom_model_path
}

# Sidebar with enhanced professional font and design
st.sidebar.markdown("""
    <style>
    /* Sidebar Styling */
    .sidebar {
        background-color: #ffffff;  /* Light background for the sidebar */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* Sidebar title style */
    .sidebar-title {
        font-size: 30px;
        font-weight: bold;
        color: #2980b9;  /* Lighter blue for the title */
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 2px solid #1abc9c;  /* Soft green underline */
        margin-bottom: 20px;
    }

    /* Sidebar description style */
    .sidebar-description {
        font-size: 16px;
        font-family: 'Arial', sans-serif;
        color: #7f8c8d;  /* Soft grey text color */
        text-align: justify;
    }

    /* Radio button label styling */
    .sidebar-radio label {
        font-size: 18px;
        font-weight: 600;
        color: #34495e;  /* Darker grey for labels */
        padding: 10px 0;
    }
    
    /* Active selection color */
    .sidebar-radio input:checked ~ label {
        color: #1abc9c;  /* Green when selected */
    }

    /* Hover effect for model options */
    .model-option {
        font-size: 16px;
        font-family: 'Arial', sans-serif;
        color: #34495e;
        padding: 10px;
    }

    .model-option:hover {
        background-color: #ecf0f1;  /* Light hover effect */
        border-radius: 5px;
        cursor: pointer;
    }

    /* Add padding for the sidebar */
    .stSidebar > div:first-child {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    </style>
    <div class="sidebar-title">Word2Vec Model Selection</div>
    <div class="sidebar-description">
        Select one of the pre-trained Word2Vec models below. You can choose between Google's large-scale pre-trained model or the custom model tailored to bbc news dataset.
    </div>
""", unsafe_allow_html=True)

model_choice = st.sidebar.radio(
    "Choose a pre-trained model to use:",
    options=list(model_options.keys())
)

# Display model details based on selection
if model_choice == "Google News Word2Vec":
    model_details = """
    **Google News Word2Vec** is a pre-trained model from Google, trained on 100 billion words from Google News dataset. 
    It contains 3 million words and phrases, and is 300-dimensional.
    """
elif model_choice == "My Custom Word2Vec Model":
    model_details = """
    **My Custom Word2Vec Model** is a Word2Vec model that is trained on the bbc news dataset. 
    It contains word vectors based on a specific data, and may have a different vocabulary and embedding dimension.
    """

# Show the model details in the main page
st.sidebar.markdown(model_details)
st.markdown(
    """
    <h1 style="text-align: center;color: #5B2C6F;">
        Word2Vec Similarity Checker
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """

    This tool calculates the cosine similarity between two words using pre-trained Word2Vec models. 
    Select a model from the sidebar and enter two words to compare their similarity.

    **How it works:**
    - Choose between **Google News Word2Vec** and **your custom model**.
    - Enter two words to calculate their similarity based on the model's word vectors.
    - The tool will display a similarity score from -1 (completely dissimilar) to 1 (highly similar).
    """
)

# Show loading spinner while the model is being loaded
with st.spinner('Loading model...'):
    model = load_model(model_options[model_choice])

if model:
    st.success(f"'{model_choice}' model loaded successfully!")
else:
    st.error(f"Failed to load '{model_choice}' model.")

if model:
    col1, col2 = st.columns(2)
    with col1:
        word1 = st.text_input("Enter the first word:", "king", max_chars=30)
    with col2:
        word2 = st.text_input("Enter the second word:", "queen", max_chars=30)

    # Word similarity result
    st.markdown("### Word Similarity Result:")
    
    if word1 and word2:
        try:
            similarity = model.similarity(word1, word2)
            st.write(f"Similarity between **'{word1}'** and **'{word2}'**: **{similarity:.4f}**")
            st.success("Similarity calculated successfully!")
        except KeyError as e:
            st.error(f"Word '{e.args[0]}' not found in the vocabulary. Please try again with a different word.")
    
    # Add a horizontal line for better section separation
    st.markdown("---")

    # Add an additional explanation
    st.markdown("""
    **More Information:**
    - The similarity score is based on the cosine similarity measure between word vectors.
    - A score close to **1** indicates that the words are very similar in meaning.
    - A score close to **-1** indicates that the words are opposite in meaning.
    - A score of **0** suggests no similarity at all.
    """)

    # Display the model's vocabulary size
    st.write(f"Vocabulary Size: {len(model.index_to_key)} words")

else:
    st.warning("Please load the model first to proceed with the similarity comparison.")