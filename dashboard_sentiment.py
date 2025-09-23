import pandas as pd
import streamlit as st
import plotly.express as px


REVIEWS_FILE = r"C:\Users\lenovo\OneDrive\Desktop\infosys\ai-batch-2\reviews_with_sentiment.csv"

@st.cache_data
def load_reviews(file_path):
    df = pd.read_csv(file_path)
    df["Review_Date"] = pd.to_datetime(df["Review_Date"], errors="coerce")
    df.dropna(subset=["Review_Text", "Sentiment", "Sentiment_Score"], inplace=True)
    return df

reviews_df = load_reviews(REVIEWS_FILE)


st.set_page_config(
    page_title="Sentiment Dashboard",
    page_icon="🛒",
    layout="wide"
)

st.title("📊 Competitor Analysis")


# Product Selection with search

products = reviews_df["Product_Name"].unique().tolist()
selected_product = st.selectbox(
    "Enter a product name or select from the list below:",
    options=[""] + products,  
    index=0
)

if selected_product:
    product_reviews = reviews_df[reviews_df["Product_Name"] == selected_product]

    
    if not product_reviews.empty and "Product_ASIN" in product_reviews.columns:
        product_asin = product_reviews["Product_ASIN"].iloc[0]
        if pd.notna(product_asin):
            product_image_url = f"https://images.amazon.com/images/P/{product_asin}.jpg"
            st.markdown(
                f"<div style='text-align: center;'><img src='{product_image_url}' width='400' alt='{selected_product}'></div>",
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ Product ASIN is missing, image not available")
    st.divider()

    
    st.header("🗣 Customer Sentiment Analysis")

    # Count sentiment labels
    sentiment_counts = product_reviews["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    # Bar chart
    fig = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_discrete_map={
            "Positive": "#69f542",
            "Neutral": "#42ddf5",
            "Negative": "#f54248"
        },
        title=f"Sentiment Distribution for {selected_product}"
    )
    st.plotly_chart(fig, use_container_width=True)

    
    # Map sentiment to intuitive score
    sentiment_map = {
        "Negative": 0,
        "Neutral": 0.5,
        "Positive": 1
    }
    product_reviews["Sentiment_Score_Intuitive"] = product_reviews["Sentiment"].map(sentiment_map)

    # Average sentiment score 
    avg_score_percentage = product_reviews["Sentiment_Score_Intuitive"].mean() * 100
    st.metric(label="Customer Sentiment Score", value=f"{avg_score_percentage:.1f}%")

    
    
    st.header("⭐ Top Reviews")

    top_reviews = pd.concat([
        product_reviews[product_reviews["Sentiment"] == "Positive"].sort_values(
            by="Sentiment_Score", ascending=False
        ).head(5),
        product_reviews[product_reviews["Sentiment"] == "Negative"].sort_values(
            by="Sentiment_Score"
        ).head(5)
    ])

    if not top_reviews.empty:
        
        display_cols = ["Sentiment", "Review_Title", "Review_Text", "Rating", "Review_Date"]
        st.dataframe(top_reviews[display_cols].reset_index(drop=True), use_container_width=True)
    else:
        st.info("No top reviews available for this product.")

    

else:
    st.info("Please select a product from the list to see analysis.")
