
import streamlit as st
import joblib
import pandas as pd
import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Download NLTK resources (only once)
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class SpamClassifier:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(
            min_df=2,
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
    def extract_features(self, text):
        """Extract additional features for spam detection"""
        text = str(text).lower()
        features = {
            'url_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
            'spam_keyword_count': sum(text.count(word) for word in ['free', 'win', 'winner', 'click', 'offer']),
            'special_char_count': sum(1 for char in text if char in string.punctuation),
            'number_count': len(re.findall(r'\d+', text)),
            'uppercase_count': sum(1 for char in text if char.isupper())
        }
        return features
    
    def preprocess_text(self, text):
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        words = [self.lemmatizer.lemmatize(word) 
                for word in text.split() 
                if word not in self.stop_words]
        return " ".join(words)

# Streamlit App
def main():
    st.set_page_config(page_title="Spam Classifier", layout="wide")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose Mode", ["Predict", "Train Model"])
    show_cool_signature()
    if app_mode == "Predict":
        predict_spam()
    else:
        train_model()

def predict_spam():
    st.title("📧 Spam Email Classifier")
    st.markdown("""
    This app predicts whether an email/message is spam or ham (legitimate).
    """)
    
    # Load model
    try:
        model_data = joblib.load('spam_model.pkl')
        model = model_data['model']
        vectorizer = model_data['vectorizer']
    except:
        st.warning("Model not found. Please train a model first.")
        return
    
    # Input form
    with st.form("prediction_form"):
        message = st.text_area("Enter your message:", height=200)
        submitted = st.form_submit_button("Predict")
        
        if submitted and message:
            # Preprocess and predict
            classifier = SpamClassifier()
            clean_text = classifier.preprocess_text(message)
            features = classifier.extract_features(message)
            
            # Vectorize text
            text_features = vectorizer.transform([clean_text]).toarray()
            additional_features = np.array([list(features.values())])
            combined_features = np.hstack([text_features, additional_features])
            
            # Make prediction
            prediction = model.predict(combined_features)[0]
            proba = model.predict_proba(combined_features)[0]
            
            # Display results
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Prediction", "SPAM 🚨" if prediction == 1 else "HAM ✅")
            
            with col2:
                st.metric("Confidence", f"{max(proba)*100:.1f}%")
            
            # Show probabilities
            st.progress(proba[1] if prediction == 1 else proba[0])
            st.write(f"Ham Probability: {proba[0]*100:.1f}%")
            st.write(f"Spam Probability: {proba[1]*100:.1f}%")
            
            # Show feature analysis
            with st.expander("Message Analysis"):
                st.write("**Extracted Features:**")
                st.json(features)
                
                st.write("**Preprocessed Text:**")
                st.code(clean_text)

def show_cool_signature():
    st.sidebar.markdown("""
        <div style='
            margin-top: 250px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #888;
            font-size: 0.75em;
            text-align: center;
            font-family: "Courier New", monospace;
        '>
            Built with code & caffeine <br>— by <strong>Sreeja ❤</strong>
        </div>
    """, unsafe_allow_html=True)

def train_model():
    st.title("🛠️ Train Spam Classification Model")
    st.markdown("Upload your training data (CSV with 'Category' and 'Message' columns)")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            if not all(col in df.columns for col in ['Category', 'Message']):
                st.error("CSV must contain 'Category' and 'Message' columns")
                return
            
            # Show sample data
            with st.expander("View Sample Data"):
                st.dataframe(df.head())
            
            # Initialize
            classifier = SpamClassifier()
            
            # Preprocess
            df['label'] = df['Category'].map({'ham': 0, 'spam': 1})
            df['clean_text'] = df['Message'].apply(classifier.preprocess_text)
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Vectorize
            status_text.text("Vectorizing text...")
            X_text = classifier.vectorizer.fit_transform(df['clean_text'])
            
            # Extract features
            status_text.text("Extracting features...")
            features = df['Message'].apply(classifier.extract_features).apply(pd.Series)
            X = np.hstack([X_text.toarray(), features.values])
            y = df['label'].values
            progress_bar.progress(30)
            
            # Balance dataset
            status_text.text("Balancing dataset...")
            from imblearn.over_sampling import SMOTE
            X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
            progress_bar.progress(50)
            
            # Train-test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_res, y_res, test_size=0.2, random_state=42)
            
            # Train models
            status_text.text("Training models...")
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            
            models = {
                'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
                'LogisticRegression': LogisticRegression(class_weight='balanced', random_state=42)
            }
            
            results = []
            best_model = None
            best_score = 0
            
            for name, model in models.items():
                status_text.text(f"Training {name}...")
                model.fit(X_train, y_train)
                
                # Evaluate
                from sklearn.metrics import classification_report, accuracy_score
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                spam_recall = ((y_test == 1) & (y_pred == 1)).sum() / (y_test == 1).sum()
                
                results.append({
                    'Model': name,
                    'Accuracy': accuracy,
                    'Spam Recall': spam_recall
                })
                
                if spam_recall > best_score:
                    best_score = spam_recall
                    best_model = model
            
            progress_bar.progress(90)
            
            # Show results
            status_text.text("Finalizing...")
            results_df = pd.DataFrame(results)
            st.subheader("Model Performance")
            st.dataframe(results_df.style.format({
                'Accuracy': '{:.2%}',
                'Spam Recall': '{:.2%}'
            }))
            
            # Confusion matrix for best model
            st.subheader("Confusion Matrix (Best Model)")
            y_pred = best_model.predict(X_test)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, display_labels=['Ham', 'Spam'], ax=ax)
            st.pyplot(fig)
            
            # Save model
            model_data = {
                'model': best_model,
                'vectorizer': classifier.vectorizer,
                'feature_names': features.columns.tolist()
            }
            
            joblib.dump(model_data, 'spam_model.pkl')
            progress_bar.progress(100)
            status_text.text("Training complete!")
            
            st.success("Model trained and saved successfully!")
            st.download_button(
                label="Download Model",
                data=open('spam_model.pkl', 'rb'),
                file_name='spam_model.pkl',
                mime='application/octet-stream'
            )
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
