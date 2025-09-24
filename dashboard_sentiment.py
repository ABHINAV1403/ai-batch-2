import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# ---------------- File Paths ----------------
REVIEWS_FILE = r"C:\Users\lenovo\OneDrive\Desktop\infosys\ai-batch-2\reviews_with_sentiment.csv"
PRODUCTS_FILE = r"C:\Users\lenovo\OneDrive\Desktop\infosys\ai-batch-2\Amazon\products.csv"

# ---------------- Load Data ----------------
@st.cache_data
def load_reviews(file_path):
    df = pd.read_csv(file_path)
    df["Review_Date"] = pd.to_datetime(df["Review_Date"], errors="coerce")
    df.dropna(subset=["Review_Text", "Sentiment", "Sentiment_Score"], inplace=True)
    return df

@st.cache_data
def load_products(file_path):
    df = pd.read_csv(file_path)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["MRP"] = pd.to_numeric(df["MRP"], errors="coerce")
    df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce")
    return df

reviews_df = load_reviews(REVIEWS_FILE)
products_df = load_products(PRODUCTS_FILE)

# ---------------- Streamlit Page Config ----------------
st.set_page_config(
    page_title="E-Commerce Competitor Strategy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {font-size:2.3rem;color:#1f77b4;text-align:center;margin-bottom:1rem;}
.section-header {font-size:1.6rem;color:#2e86ab;margin-top:2rem;margin-bottom:1rem;}
.positive-sentiment { color:#28a745; }
.negative-sentiment { color:#dc3545; }
.neutral-sentiment  { color:#ffc107; }
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
section = st.sidebar.radio("Navigate", ["Product Analysis", "Competitor Comparison", "Strategic Recommendations"])
products = products_df["Product_Name"].unique().tolist()
selected_product = st.sidebar.selectbox("Select Product", products)

# ---------------- Product Analysis ----------------
if section == "Product Analysis":
    st.markdown('<div class="main-header">Product Analysis</div>', unsafe_allow_html=True)

    prod = products_df[products_df["Product_Name"] == selected_product].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"₹{int(prod['Price'])}")
    c2.metric("MRP", f"₹{int(prod['MRP'])}")
    c3.metric("Discount", f"{prod['Discount']:.1f}%")

    # Product image
    if pd.notna(prod["Product_ASIN"]):
        product_image_url = f"https://images.amazon.com/images/P/{prod['Product_ASIN']}.jpg"
        st.markdown(
            f"<div style='text-align: center;'><img src='{product_image_url}' width='300' alt='{selected_product}'></div>",
            unsafe_allow_html=True
        )

    # Sentiment analysis
    product_reviews = reviews_df[reviews_df["Product_Name"] == selected_product]
    if not product_reviews.empty:
        st.header("🗣 Customer Sentiment Analysis")

        sentiment_counts = product_reviews["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]

        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            color_discrete_map={"Positive": "#69f542", "Neutral": "#42ddf5", "Negative": "#f54248"},
            title=f"Sentiment Distribution for {selected_product}"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Map sentiment to intuitive score
        sentiment_map = {"Negative": 0, "Neutral": 0.5, "Positive": 1}
        product_reviews["Sentiment_Score_Intuitive"] = product_reviews["Sentiment"].map(sentiment_map)
        avg_score_percentage = product_reviews["Sentiment_Score_Intuitive"].mean() * 100
        st.metric(label="Customer Sentiment Score", value=f"{avg_score_percentage:.1f}%")

        # Top reviews
        st.header("⭐ Top Reviews")
        top_reviews = pd.concat([
            product_reviews[product_reviews["Sentiment"] == "Positive"].sort_values(by="Sentiment_Score", ascending=False).head(5),
            product_reviews[product_reviews["Sentiment"] == "Negative"].sort_values(by="Sentiment_Score").head(5)
        ])
        display_cols = ["Sentiment", "Review_Title", "Review_Text", "Rating", "Review_Date"]
        st.dataframe(top_reviews[display_cols].reset_index(drop=True), use_container_width=True)
    else:
        st.info("No reviews available for this product.")

# ---------------- Competitor Comparison ----------------
elif section == "Competitor Comparison":
    st.markdown('<div class="main-header">Competitor Comparison</div>', unsafe_allow_html=True)
    
    # Scatter Plot + Regression: Price vs Sentiment
    st.subheader("Price vs Sentiment Analysis")
    
    # Compute average sentiment per product
    sentiment_map = {"Negative": 0, "Neutral": 0.5, "Positive": 1}
    reviews_df["Sentiment_Score_Intuitive"] = reviews_df["Sentiment"].map(sentiment_map)
    avg_sentiment = reviews_df.groupby("Product_Name")["Sentiment_Score_Intuitive"].mean().reset_index()
    
    merged_df = products_df.merge(avg_sentiment, left_on="Product_Name", right_on="Product_Name", how="left").dropna(subset=["Sentiment_Score_Intuitive"])
    
    # Scatter plot using seaborn
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=merged_df, x="Price", y="Sentiment_Score_Intuitive", s=100, color="blue", alpha=0.7)
    
    # Regression line
    X = merged_df["Price"].values.reshape(-1, 1)
    y = merged_df["Sentiment_Score_Intuitive"].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    plt.plot(merged_df["Price"], y_pred, color="red", linewidth=2, label="Trend Line")
    
    plt.xlabel("Price (INR)")
    plt.ylabel("Sentiment Score")
    plt.title("Price vs Sentiment Trend Across Products")
    plt.legend()
    st.pyplot(plt)
    
    # Correlation
    corr = merged_df["Price"].corr(merged_df["Sentiment_Score_Intuitive"])
    st.write(f"Pearson correlation between Price and Sentiment: {corr:.2f}")
    
    # Highlight Excellent Products
    st.subheader("Excellent Products (Low Price + High Sentiment)")
    low_price = merged_df["Price"].quantile(0.3)
    high_sentiment = merged_df["Sentiment_Score_Intuitive"].quantile(0.7)
    excellent = merged_df[(merged_df["Price"] <= low_price) & (merged_df["Sentiment_Score_Intuitive"] >= high_sentiment)]
    st.dataframe(excellent[["Product_Name","Price","Sentiment_Score_Intuitive"]].reset_index(drop=True))
    
    # Optional: Original competitor price bar
    st.subheader("Competitor Price Comparison")
    fig = px.bar(products_df, x="Product_Name", y="Price", color="Price", title="Competitor Price Comparison")
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(products_df[["Product_Name", "Price", "MRP", "Discount", "Rating"]])

# ---------------- Strategic Recommendations ----------------
else:
    st.markdown('<div class="main-header">Strategic Recommendations</div>', unsafe_allow_html=True)
    product_reviews = reviews_df[reviews_df["Product_Name"] == selected_product]
    avg_score = 0
    if not product_reviews.empty:
        sentiment_map = {"Negative": 0, "Neutral": 0.5, "Positive": 1}
        product_reviews["Sentiment_Score_Intuitive"] = product_reviews["Sentiment"].map(sentiment_map)
        avg_score = product_reviews["Sentiment_Score_Intuitive"].mean()

    if avg_score < 0.4:
        st.markdown("### ⚠ Needs Improvement")
        st.write("- Address negative reviews quickly\n- Offer discounts or bundles to attract buyers\n- Improve product quality if recurring complaints appear")
    elif avg_score < 0.7:
        st.markdown("### 🙂 Good Standing")
        st.write("- Maintain competitive pricing\n- Run limited-time promotions\n- Encourage happy customers to leave reviews")
    else:
        st.markdown("### 🌟 Excellent Performance")
        st.write("- Keep product quality consistent\n- Use positive reviews in marketing campaigns\n- Explore premium pricing strategies")
