import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
import os

# Initialize session state
if "user_submitted" not in st.session_state:
    st.session_state.user_submitted = False

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("travel_data.csv")
    df["Popularity"] = pd.to_numeric(df["Popularity"], errors="coerce")
    return df

df = load_data()

# Machine Learning Model
def train_knn_model(df):
    # Drop rows with missing values
    df_filtered = df.dropna(subset=["Budget (INR)", "Duration (days)", "Rating", 
                                    "Latitude", "Longitude", "Climate", "Best Season", "Type", "Destination"])

    # Label encode categorical features
    le_climate = LabelEncoder()
    le_season = LabelEncoder()
    le_type = LabelEncoder()

    df_filtered["Climate_enc"] = le_climate.fit_transform(df_filtered["Climate"])
    df_filtered["Season_enc"] = le_season.fit_transform(df_filtered["Best Season"])
    df_filtered["Type_enc"] = le_type.fit_transform(df_filtered["Type"])

    # Balance the dataset by downsampling
    min_count = df_filtered["Destination"].value_counts().min()
    balanced_df = df_filtered.groupby("Destination").apply(lambda x: x.sample(min_count)).reset_index(drop=True)

    # Features
    features = ["Budget (INR)", "Duration (days)", "Rating", "Latitude", "Longitude",
                "Climate_enc", "Season_enc", "Type_enc"]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(balanced_df[features])

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(scaled_features, balanced_df["Destination"])

    return knn, scaler, balanced_df, le_climate, le_season, le_type

knn_model, scaler, df_filtered, le_climate, le_season, le_type = train_knn_model(df)

# Styling
st.markdown("""
    <style>
    .main-title { text-align: center; font-size: 36px; color: #ff5733; }
    .sidebar-header { font-size: 20px; color: #673ab7; }
    .stButton>button { background-color: #ff9800; color: white; }
    </style>
""", unsafe_allow_html=True)

# If user hasn't submitted details
if not st.session_state.user_submitted:
    st.markdown('<h1 class="main-title">📝 User Details</h1>', unsafe_allow_html=True)
    st.markdown("Fill in your details to get personalized travel recommendations!")

    name = st.text_input("👤 Name:")
    email = st.text_input("📧 Email:")
    age = st.number_input("🎂 Age:", min_value=10, max_value=100, value=25)
    preferred_travel = st.selectbox("🌍 Preferred Travel Type:", df["Type"].dropna().unique())

    if st.button("Submit"):
        user_data = pd.DataFrame([[name, email, age, preferred_travel]], 
                                 columns=["Name", "Email", "Age", "Preferred Travel"])
        if os.path.exists("user_data.xlsx"):
            existing_data = pd.read_excel("user_data.xlsx")
            user_data = pd.concat([existing_data, user_data], ignore_index=True)
        user_data.to_excel("user_data.xlsx", index=False)
        st.session_state.user_submitted = True
        st.rerun()

# If user has submitted details
else:
    st.markdown('<h1 class="main-title">🌍 AI-Based Travel Recommendation System</h1>', unsafe_allow_html=True)
    st.image("https://source.unsplash.com/1600x900/?travel", use_container_width=True)
    st.markdown("Discover your next dream destination with AI-powered recommendations!")

    st.sidebar.markdown('<h2 class="sidebar-header">📌 Customize Your Trip</h2>', unsafe_allow_html=True)
    travel_type = st.sidebar.selectbox("Select Travel Type:", df["Type"].dropna().unique())
    budget = st.sidebar.slider("Select Maximum Budget (INR):", 5000, 50000, 20000, 5000)
    duration = st.sidebar.slider("Select Duration (days):", 1, 10, 5)
    rating = st.sidebar.slider("Select Minimum Rating (1-5):", 1.0, 5.0, 4.0, 0.1)
    climate = st.sidebar.selectbox("Preferred Climate:", df["Climate"].dropna().unique())
    popularity = st.sidebar.selectbox("Popularity:", df["Popularity"].dropna().unique())
    season = st.sidebar.selectbox("Preferred Travel Season:", df["Best Season"].dropna().unique())
    latitude = st.sidebar.number_input("Your Current Latitude", value=28.7041)
    longitude = st.sidebar.number_input("Your Current Longitude", value=77.1025)

    if st.sidebar.button("Get AI Recommendation"):
        # Encode categorical inputs
        climate_enc = le_climate.transform([climate])[0]
        season_enc = le_season.transform([season])[0]
        type_enc = le_type.transform([travel_type])[0]

        # Prepare input
        input_data = np.array([[budget, duration, rating, latitude, longitude, 
                                climate_enc, season_enc, type_enc]])
        input_scaled = scaler.transform(input_data)

        # Top 3 Recommendations
        probs = knn_model.predict_proba(input_scaled)
        top_indices = probs[0].argsort()[-3:][::-1]
        top_destinations = knn_model.classes_[top_indices]

        st.success(f"✨ **Top Recommendations:** {', '.join(top_destinations)}")

        # Show details for top destination
        top_place = top_destinations[0]
        recommended_data = df_filtered[df_filtered["Destination"] == top_place]
        st.write(recommended_data)

        if not recommended_data.empty:
            st.subheader("📍 Recommended Destination on Map")
            map_center = [recommended_data.iloc[0]["Latitude"], recommended_data.iloc[0]["Longitude"]]
            travel_map = folium.Map(location=map_center, zoom_start=5)
            folium.Marker(
                location=map_center,
                popup=f"<b>{top_place}</b><br>Budget: ₹{budget}<br>Rating: {rating}",
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(travel_map)
            folium_static(travel_map)

    # Reset
    if st.sidebar.button("🔄 Enter New User"):
        st.session_state.user_submitted = False
        st.rerun()

# Footer
st.markdown("---")
st.markdown("🚀 *MADE BY: TUSHAR MAHAJAN & KAVYA PODDAR* | 📩 *Contact us for more personalized travel suggestions!*")
