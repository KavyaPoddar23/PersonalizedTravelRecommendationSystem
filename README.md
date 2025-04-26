# AI-Based Travel Recommendation System

An intelligent travel recommendation system that uses machine learning to suggest personalized travel destinations based on user preferences.

## Features

- User-friendly interface with Streamlit
- AI-powered destination recommendations using K-Nearest Neighbors algorithm
- Interactive map visualization of recommended destinations
- Customizable travel preferences (budget, duration, climate, etc.)
- User data collection and management
- Real-time recommendations based on multiple parameters

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run TravelRecommendation.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Follow these steps:
   - Enter your user details (name, email, age, preferred travel type)
   - Customize your travel preferences in the sidebar
   - Click "Get AI Recommendation" to receive personalized destination suggestions
   - View the recommended destinations on an interactive map
   - Use the "Enter New User" button to start over

## Data Files

- `travel_data.csv`: Contains the travel destination dataset
- `user_data.xlsx`: Stores user information and preferences

## Project Structure

- `TravelRecommendation.py`: Main application file
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation

## Authors

- Tushar Mahajan
- Kavya Poddar

## License

This project is licensed under the MIT License - see the LICENSE file for details. 