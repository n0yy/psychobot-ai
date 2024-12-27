import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.model import load, predict
from src.llm import make_prompt, get_summary

__version__ = "0.0.1"

# Load the model
@st.cache_resource
def load_model():
    return load("./model/lgbm.pkl")

def create_emotion_chart(chart_data):
    """Create a bar chart for emotion probabilities"""
    if isinstance(chart_data, pd.DataFrame) and "Emotional" in chart_data.columns and "Probability" in chart_data.columns:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=chart_data["Emotional"],
            y=chart_data["Probability"],
            marker_color='rgb(26, 118, 255)'
        ))
        
        fig.update_layout(
            title="Emotional Analysis Distribution",
            xaxis_title="Emotion",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            template="simple_white",
            showlegend=False
        )
        return fig
    return None

def main():
    # Page configuration
    st.set_page_config(
        page_title="PsychoBot AI",
        page_icon="🤖",
        layout="wide"
    )
    
    # Title and subtitle with custom styling
    st.title("🤖 PsychoBot AI: Social Media Behavior Analyzer")
    st.subheader("Unlock insights into social media usage patterns and their potential psychological impacts. Our AI-powered analysis helps you understand digital behavior better.")
    
    # Version info
    st.sidebar.text(f"Version: {__version__}")
    
    # Create form
    with st.form("prediction_form"):
        st.subheader("📝 Enter User Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=13, max_value=100, value=25)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            platform = st.selectbox("Platform", ["Instagram", "Facebook", "Twitter", "TikTok", "LinkedIn"])
            daily_usage = st.number_input("Daily Usage Time (minutes)", min_value=0, max_value=1440, value=120)
        
        with col2:
            posts = st.number_input("Posts Per Day", min_value=0, max_value=100, value=2)
            likes = st.number_input("Likes Received Per Day", min_value=0, value=50)
            comments = st.number_input("Comments Received Per Day", min_value=0, value=10)
            messages = st.number_input("Messages Sent Per Day", min_value=0, value=20)
        
        submit_button = st.form_submit_button("Analyze")
    
    if submit_button:
        # Show loading spinner
        with st.spinner("Analyzing social media behavior..."):
            # Prepare data
            data = {
                "Age": [float(age)],
                "Gender": [gender],
                "Platform": [platform],
                "Daily_Usage_Time (minutes)": [float(daily_usage)],
                "Posts_Per_Day": [float(posts)],
                "Likes_Received_Per_Day": [float(likes)],
                "Comments_Received_Per_Day": [float(comments)],
                "Messages_Sent_Per_Day": [float(messages)]
            }
            
            df = pd.DataFrame(data)
            model = load_model()
            
            try:
                # Get prediction and chart data
                prediction, chart_data = predict(df, model)
                
                # Verify chart_data structure
                if not isinstance(chart_data, pd.DataFrame):
                    st.error("Invalid chart data format received from model")
                    return
                prompt = make_prompt(chart_data)
                summary = get_summary(prompt)
                
                # Display results
                st.success("Analysis Complete!")
                
                # Create columns for results
                result_col1, result_col2 = st.columns([2, 1])

                chart_data["Probability"] = chart_data["Probability"].astype(float)
                
                with result_col1:
                    st.subheader("📊 Analysis Results")
                    # Display the summary
                    st.markdown(summary)
                    
                    # Create and display chart
                    fig = create_emotion_chart(chart_data)
                    if fig:
                        st.plotly_chart(fig)
                    else:
                        st.warning("Unable to generate visualization due to data format issues")
                
                with result_col2:
                    st.subheader("🎯 Prediction Score")
                    # Display the prediction in a metric
                    st.dataframe(chart_data)
                    
                    # Additional insights box
                    st.info("💡 Quick Insights", icon="💡")
                    engagement_rate = (likes + comments) / posts if posts > 0 else 0
                    st.write(f"""
                    - Platform: {platform}
                    - Daily Usage: {daily_usage} minutes
                    - Engagement Rate: {engagement_rate:.1f}
                    """)
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.write("Please check your input data and try again.")

if __name__ == "__main__":
    main()