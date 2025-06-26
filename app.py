import streamlit as st
import joblib
import re
import pandas as pd
import os
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and vectorizer
@st.cache_resource
def load_model():
    """Load the trained model and vectorizer"""
    try:
        model = joblib.load("model/model.pkl")
        vectorizer = joblib.load("model/vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found! Please make sure you have trained the model first by running train_model.py")
        st.stop()

def clean_text(text):
    """Clean and preprocess text data - same as in training"""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)     # remove URLs
    text = re.sub(r"\d+", "", text)         # remove numbers
    text = re.sub(r"[^\w\s]", "", text)     # remove punctuation
    return text.strip()

def predict_news(text, model, vectorizer):
    """Make prediction on input text"""
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Vectorize the text
    text_vec = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]
    
    return prediction, probability

def analyze_text_features(text):
    """Analyze text for common fake news indicators"""
    text_lower = text.lower()
    
    # Fake news indicators
    fake_indicators = []
    if any(word in text_lower for word in ["breaking", "urgent", "alert"]):
        fake_indicators.append("Sensationalist headlines")
    if any(word in text_lower for word in ["shocking", "unbelievable", "incredible"]):
        fake_indicators.append("Emotional language")
    if any(phrase in text_lower for phrase in ["government cover", "conspiracy", "they don't want you to know"]):
        fake_indicators.append("Conspiracy language")
    if any(word in text_lower for word in ["anonymous", "secret source", "insider"]):
        fake_indicators.append("Unverifiable sources")
    if "!!!" in text or text.count("!") > 3:
        fake_indicators.append("Excessive exclamation marks")
    
    # Real news indicators
    real_indicators = []
    if any(word in text_lower for word in ["according to", "study shows", "research indicates"]):
        real_indicators.append("Attribution to sources")
    if any(word in text_lower for word in ["dr.", "professor", "researcher"]):
        real_indicators.append("Expert sources")
    if any(word in text_lower for word in ["university", "institute", "journal", "published"]):
        real_indicators.append("Academic/institutional sources")
    if any(word in text_lower for word in ["data", "statistics", "study", "research"]):
        real_indicators.append("Evidence-based language")
    
    return fake_indicators, real_indicators

def main():
    # Title and description
    st.title("📰 Fake News Detector")
    st.markdown("---")
    
    st.markdown("""
    This app uses machine learning to detect whether a news article is **real** or **fake**.
    Simply paste your news article text below and get instant predictions!
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model, vectorizer = load_model()
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Model Details:**
        - Algorithm: Logistic Regression
        - Features: TF-IDF Vectorization
        - Max Features: 5,000
        - Training Data: Real + Fake news articles
        
        **How it works:**
        1. Text is cleaned and preprocessed
        2. Converted to numerical features using TF-IDF
        3. Model predicts probability of being real/fake
        """)
        
        # Load evaluation metrics if available
        if os.path.exists("model/evaluation.txt"):
            st.header("📊 Model Performance")
            with open("model/evaluation.txt", "r") as f:
                eval_text = f.read()
            st.text(eval_text)
    
    # Main input area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Enter News Article")
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["Type/Paste Text", "Upload Text File"],
            horizontal=True
        )
        
        article_text = ""
        
        if input_method == "Type/Paste Text":
            article_text = st.text_area(
                "Paste your news article here:",
                height=300,
                placeholder="Enter the title and content of the news article you want to analyze..."
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a text file:",
                type=['txt'],
                help="Upload a .txt file containing the news article"
            )
            if uploaded_file is not None:
                article_text = uploaded_file.read().decode('utf-8')
                st.text_area("Uploaded content:", article_text, height=200)
        
        # Predict button
        if st.button("🔍 Analyze Article", type="primary", use_container_width=True):
            if article_text.strip():
                with st.spinner("Analyzing..."):
                    prediction, probability = predict_news(article_text, model, vectorizer)
                    
                    # Store results in session state
                    st.session_state.last_prediction = prediction
                    st.session_state.last_probability = probability
                    st.session_state.last_text = article_text
            else:
                st.warning("Please enter some text to analyze!")
    
    with col2:
        st.header("📊 Prediction Results")
        
        # Show results if available
        if hasattr(st.session_state, 'last_prediction'):
            prediction = st.session_state.last_prediction
            probability = st.session_state.last_probability
            
            # Determine result
            if prediction == 1:
                result = "REAL NEWS"
                confidence = probability[1] * 100
                color = "green"
                icon = "✅"
            else:
                result = "FAKE NEWS"
                confidence = probability[0] * 100
                color = "red"
                icon = "❌"
            
            # Display result with styling
            st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: {color}; color: white; text-align: center; margin: 10px 0;'>
                <h2>{icon} {result}</h2>
                <h3>Confidence: {confidence:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence breakdown
            st.subheader("Confidence Breakdown:")
            
            # Create probability bars
            fake_prob = probability[0] * 100
            real_prob = probability[1] * 100
            
            st.markdown("**Fake News Probability:**")
            st.progress(fake_prob / 100)
            st.markdown(f"<span style='color: red;'>{fake_prob:.1f}%</span>", unsafe_allow_html=True)
            
            st.markdown("**Real News Probability:**")
            st.progress(real_prob / 100)
            st.markdown(f"<span style='color: green;'>{real_prob:.1f}%</span>", unsafe_allow_html=True)
            
            # Additional insights
            st.subheader("💡 Insights:")
            if confidence > 80:
                st.success("High confidence prediction")
            elif confidence > 60:
                st.warning("Moderate confidence - consider additional verification")
            else:
                st.error("Low confidence - manual verification recommended")
            
            # Text analysis section
            with st.expander("🔍 Text Analysis Details"):
                fake_indicators, real_indicators = analyze_text_features(st.session_state.last_text)
                
                col_debug1, col_debug2 = st.columns(2)
                
                with col_debug1:
                    st.markdown("**📊 Text Statistics:**")
                    word_count = len(st.session_state.last_text.split())
                    char_count = len(st.session_state.last_text)
                    st.metric("Word Count", word_count)
                    st.metric("Character Count", char_count)
                
                with col_debug2:
                    st.markdown("**🔧 Preprocessed Text Preview:**")
                    cleaned_preview = clean_text(st.session_state.last_text[:150] + "...")
                    st.code(cleaned_preview, language=None)
                
                if fake_indicators:
                    st.markdown("**🚨 Potential Fake News Indicators:**")
                    for indicator in fake_indicators:
                        st.markdown(f"- ❌ {indicator}")
                
                if real_indicators:
                    st.markdown("**✅ Potential Real News Indicators:**")
                    for indicator in real_indicators:
                        st.markdown(f"- ✅ {indicator}")
                
                if not fake_indicators and not real_indicators:
                    st.markdown("**⚪ No obvious language indicators found. Model relies on learned TF-IDF patterns.**")
                
        else:
            st.info("Enter a news article and click 'Analyze Article' to see results here.")
    
    # Example texts section
    st.markdown("---")
    st.header("📝 Try These Examples")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Example Real News")
        if st.button("Load Real News Example"):
            example_real = """
            COVID-19 Vaccine Development Shows Promising Results in Phase 3 Trials
            
            BOSTON - Researchers at Massachusetts General Hospital announced today that their COVID-19 vaccine candidate has shown 94.5% efficacy in preventing severe illness during Phase 3 clinical trials involving 30,000 participants across multiple countries.
            
            The study, published in the New England Journal of Medicine, followed participants for six months and found that only 11 cases of severe COVID-19 occurred in the vaccinated group compared to 185 cases in the placebo group. Dr. Sarah Chen, lead researcher and professor of immunology at Harvard Medical School, stated that the results exceed expectations.
            
            "These findings represent a significant milestone in our fight against the pandemic," said Dr. Chen during a press conference at the hospital. The research was conducted in collaboration with the National Institutes of Health and funded through Operation Warp Speed.
            
            The vaccine uses messenger RNA technology similar to existing vaccines and requires two doses administered 21 days apart. Common side effects include mild fatigue and soreness at the injection site, occurring in approximately 20% of participants.
            
            The pharmaceutical company plans to submit emergency use authorization to the FDA within the next two weeks. If approved, initial doses will be distributed to healthcare workers and high-risk populations by early next month.
            """
            st.session_state.example_text = example_real
            st.rerun()
    
    with col2:
        st.subheader("Example Fake News")
        if st.button("Load Fake News Example"):
            example_fake = """
            BREAKING: Secret Government Documents Reveal COVID Vaccines Contain Mind Control Chips!!!
            
            Shocking leaked documents from an anonymous whistleblower inside the CDC have revealed that COVID-19 vaccines secretly contain microscopic tracking chips designed to control people's thoughts and monitor their every move.
            
            The classified files, obtained exclusively by our investigative team, show that the government has been working with Big Pharma to implement a massive surveillance program. According to insider sources who cannot be named for their safety, the chips are activated by 5G cell towers and can influence human behavior.
            
            "This is the biggest conspiracy in human history," claims Dr. Michael Johnson, a self-proclaimed medical expert who refused to provide his credentials. "They're using the pandemic as cover to inject everyone with mind control technology."
            
            Multiple witnesses report experiencing strange symptoms after vaccination, including hearing voices and feeling compelled to buy certain products. However, mainstream media refuses to cover this story because they are controlled by the same forces behind the conspiracy.
            
            The government has denied these allegations, but their refusal to allow independent testing of vaccine contents only proves they have something to hide. Wake up, people - this is about control, not health!
            """
            st.session_state.example_text = example_fake
            st.rerun()
    
    # Show example text if selected
    if hasattr(st.session_state, 'example_text'):
        st.text_area("Example Text:", st.session_state.example_text, height=150, key="example_display")
        if st.button("Analyze Example", key="analyze_example"):
            with st.spinner("Analyzing example..."):
                prediction, probability = predict_news(st.session_state.example_text, model, vectorizer)
                st.session_state.last_prediction = prediction
                st.session_state.last_probability = probability
                st.session_state.last_text = st.session_state.example_text
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
        Made with ❤️ using Streamlit | Remember: Always verify news from multiple reliable sources!
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()