import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime, timedelta
import os
import sys
from typing import List, Dict, Any
import uuid

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Citizen AI Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# Functions to interact with the API
def get_ai_response(message: str) -> str:
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "message": message,
                "session_id": st.session_state.session_id,
                "context": [m for m in st.session_state.messages if "content" in m]
            }
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "Sorry, I couldn't process your request.")
    except Exception as e:
        st.error(f"Error communicating with AI service: {str(e)}")
        return "Sorry, I'm having trouble connecting to the server right now."

def submit_feedback(text: str, category: str = None) -> Dict[str, Any]:
    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json={"text": text, "category": category}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error submitting feedback: {str(e)}")
        return {}

def get_all_feedback() -> List[Dict[str, Any]]:
    try:
        response = requests.get(f"{API_URL}/feedback")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching feedback data: {str(e)}")
        return []

def get_sentiment_summary() -> Dict[str, int]:
    try:
        response = requests.get(f"{API_URL}/sentiment/summary")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching sentiment summary: {str(e)}")
        return {"positive": 0, "neutral": 0, "negative": 0, "total": 0}

# Sidebar navigation
st.sidebar.title("Citizen AI Platform")
page = st.sidebar.radio("Navigation", ["Chat Assistant", "Submit Feedback", "Dashboard"])

# Chat Assistant Page
if page == "Chat Assistant":
    st.header("🤖 AI Citizen Assistant")
    st.markdown("Ask questions about government services, policies, or get help with civic issues.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("How can I help you today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_ai_response(prompt)
                st.write(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Submit Feedback Page
elif page == "Submit Feedback":
    st.header("📝 Submit Your Feedback")
    st.markdown("Share your thoughts, suggestions, or concerns about government services.")
    
    # Feedback form
    with st.form("feedback_form"):
        feedback_text = st.text_area("Your Feedback", height=150)
        category = st.selectbox(
            "Category",
            ["General", "Transportation", "Healthcare", "Education", "Public Safety", "Environment", "Other"]
        )
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted and feedback_text:
            result = submit_feedback(feedback_text, category)
            if result:
                st.session_state.feedback_submitted = True
                st.success("Thank you for your feedback! Your input helps us improve our services.")
                
                # Show sentiment analysis result
                sentiment = result.get("sentiment", "")
                score = result.get("score", 0)
                
                sentiment_color = "#28a745" if sentiment == "positive" else "#ffc107" if sentiment == "neutral" else "#dc3545"
                st.markdown(f"<div style='padding: 10px; background-color: {sentiment_color}; color: white; border-radius: 5px;'>"
                          f"<strong>Sentiment Analysis:</strong> {sentiment.title()} (Score: {score:.2f})</div>", unsafe_allow_html=True)

# Dashboard Page
elif page == "Dashboard":
    st.header("📊 Citizen Feedback Dashboard")
    st.markdown("Visualizing citizen sentiment and feedback trends.")
    
    # Get data
    feedback_data = get_all_feedback()
    sentiment_summary = get_sentiment_summary()
    
    if not feedback_data:
        st.info("No feedback data available yet. Submit some feedback to see the dashboard.")
    else:
        # Convert to DataFrame
        df = pd.DataFrame(feedback_data)
        
        # Add date column from timestamp
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        # Dashboard layout with columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            
            # Prepare data for pie chart
            sentiment_counts = {
                "Positive": sentiment_summary.get("positive", 0),
                "Neutral": sentiment_summary.get("neutral", 0),
                "Negative": sentiment_summary.get("negative", 0)
            }
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = ['#28a745', '#ffc107', '#dc3545']
            wedges, texts, autotexts = ax.pie(
                sentiment_counts.values(),
                labels=sentiment_counts.keys(),
                autopct='%1.1f%%',
                startangle=90,
                colors=colors
            )
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')
            plt.setp(autotexts, size=10, weight="bold", color="white")
            plt.setp(texts, size=12, weight="bold")
            st.pyplot(fig)
            
            # Display counts
            st.markdown(f"**Total Feedback:** {sentiment_summary.get('total', 0)}")
            st.markdown(f"**Positive:** {sentiment_summary.get('positive', 0)}")
            st.markdown(f"**Neutral:** {sentiment_summary.get('neutral', 0)}")
            st.markdown(f"**Negative:** {sentiment_summary.get('negative', 0)}")
        
        with col2:
            st.subheader("Sentiment Trends Over Time")
            
            # Group by date and sentiment, count occurrences
            if len(df) > 0 and 'date' in df.columns and 'sentiment' in df.columns:
                sentiment_by_date = df.groupby(['date', 'sentiment']).size().reset_index(name='count')
                
                # Pivot the data for plotting
                pivot_data = sentiment_by_date.pivot(index='date', columns='sentiment', values='count').fillna(0)
                
                # Ensure all sentiments are represented
                for sentiment in ['positive', 'neutral', 'negative']:
                    if sentiment not in pivot_data.columns:
                        pivot_data[sentiment] = 0
                
                # Plot the data
                fig, ax = plt.subplots(figsize=(10, 6))
                pivot_data.plot(kind='line', marker='o', ax=ax)
                
                plt.title('Sentiment Trends Over Time')
                plt.xlabel('Date')
                plt.ylabel('Number of Feedback')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(title='Sentiment')
                
                st.pyplot(fig)
            else:
                st.info("Not enough data to show sentiment trends over time.")
        
        # Category distribution
        st.subheader("Feedback by Category")
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
            plt.title('Feedback by Category')
            plt.xlabel('Category')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
        
        # Recent feedback
        st.subheader("Recent Feedback")
        if not df.empty:
            recent_df = df.sort_values('timestamp', ascending=False).head(5)
            for _, row in recent_df.iterrows():
                sentiment_color = "#28a745" if row['sentiment'] == "positive" else "#ffc107" if row['sentiment'] == "neutral" else "#dc3545"
                st.markdown(f"<div style='padding: 10px; margin-bottom: 10px; border-left: 5px solid {sentiment_color}; background-color: #f8f9fa;'>"
                          f"<strong>Category:</strong> {row['category']}<br>"
                          f"<strong>Sentiment:</strong> {row['sentiment'].title()}<br>"
                          f"<strong>Feedback:</strong> {row['text']}<br>"
                          f"<small>Submitted on {row['timestamp'][:10]}</small>"
                          f"</div>", unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Citizen AI Platform")