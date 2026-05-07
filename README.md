# Stress Predictor

A Flask-based AI-powered stress prediction application using machine learning.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train the model:**
```bash
python train_model.py
```

3. **Run the application:**
```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## 📁 Project Structure

```
stress_predictor/
├── app.py                    # Flask web server and API
├── train_model.py            # ML model training script
├── index.html                # Static web interface
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── assets/                   # Asset files
├── css/
│   └── style.css            # Custom stylesheets
├── data/
│   └── stress_data.csv      # Training dataset
├── js/
│   └── app.js               # Frontend JavaScript
├── models/
│   └── stress_model.pkl     # Trained ML model
├── static/css/
│   ├── bootstrap.min.css    # Bootstrap framework
│   ├── style.css            # Bootstrap styles
│   └── tailwind.css         # Tailwind CSS
└── templates/
    └── index.html           # Flask template
```

## 💻 Technologies

- **Backend:** Flask, Python, Scikit-learn, Pandas
- **Frontend:** HTML5, Bootstrap 5, CSS3, JavaScript
- **ML:** Scikit-learn, Joblib

## 🎯 Features

- 🤖 ML-based stress prediction
- 📊 Multi-factor health analysis
- 🎨 Responsive Bootstrap UI
- 📱 Mobile-friendly design
- 💡 Personalized recommendations
- 🔌 REST API endpoint

## 📝 Input Parameters

The model analyzes six key factors:
- Sleep Quality (1-5 scale)
- Daily Exercise (minutes)
- Hours Worked (per week)
- Social Interactions (per week)
- Workload Score (1-10 scale)
- Heart Rate (BPM)

## 🔌 API Endpoint

**POST /predict**

Request:
```json
{
    "sleep_quality": 7,
    "exercise_minutes": 30,
    "hours_worked": 8,
    "social_interactions": 5,
    "work_load_score": 6,
    "heart_rate": 70
}
```

Response:
```json
{
    "prediction": 0
}
```

Values: 0 = Low Stress, 1 = Moderate Stress, 2 = High Stress

## 📚 Learn More

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Bootstrap 5 Documentation](https://getbootstrap.com/docs/5.0/)

## ⚠️ Disclaimer

This tool is for educational purposes only and is not a substitute for professional medical advice. Always consult healthcare professionals for stress assessment and treatment.

## 📄 License

Educational Use Only

## 👤 Author

Akhil Balarushi
- GitHub: [@akhilbalarushi-a11y](https://github.com/akhilbalarushi-a11y)