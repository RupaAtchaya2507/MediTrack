# 💊 MediTrack — AI-Powered Medicine Inventory & Expiry Prioritization

> A smart medicine inventory management system that uses **Machine Learning (Decision Tree Classifier)** to predict the priority status of medicines — helping users identify items that need expiry checks, restocking, or no immediate action.

---

## 📌 Table of Contents
- [About the Project](#-about-the-project)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [ML Model Details](#-ml-model-details)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Screenshots](#-screenshots)
- [Disclaimer](#-disclaimer)

---

## 📖 About the Project

MediTrack is a **Foundations of AI (FAI) case study project** that demonstrates the real-world application of a supervised machine learning algorithm in the healthcare/inventory domain.

The core idea is simple:
- You have a home or clinic medicine inventory
- Medicines can expire, run out of stock, or be perfectly fine
- Instead of manually checking each item, MediTrack uses a **trained Decision Tree model** to automatically classify each medicine into one of three priority categories

This project combines a **Flask web backend**, a **trained ML model**, and a **clean interactive UI** to deliver a fully functional prototype.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🏠 **Dashboard** | Overview of inventory with metric cards, charts, and model performance |
| ➕ **Add Medicine** | Form to add a new medicine — automatically predicted and added to inventory |
| 🔮 **Predictions Page** | Shows AI predictions for all 300+ medicines with filter by priority & category |
| 🔍 **Medicine Search** | Search any medicine by name or ID and instantly see its status |
| 🔄 **Refresh Predictions** | Re-runs the ML model on all inventory items with animated feedback |
| 📦 **View Inventory** | Full inventory table with search, filter, color-coded expiry and stock levels |
| 📊 **Model Insights** | Accuracy, Precision, Recall, F1 Score, Confusion Matrix, Feature Importance chart |
| ℹ️ **About Page** | Project details, tech stack, and disclaimer |
| 🔐 **Login / Signup** | Session-based authentication (demo mode — any credentials work) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3, Flask |
| **Machine Learning** | scikit-learn (DecisionTreeClassifier) |
| **Data Processing** | pandas |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Charts** | Chart.js |
| **Fonts** | Google Fonts — Inter, Poppins |

---

## 🤖 ML Model Details

### Algorithm
**Decision Tree Classifier** from `scikit-learn`

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
```

### Dataset
- **300 medicine records** (`medicine_inventory_expiry_synthetic.csv`)
- **80/20 train-test split** → 240 training records, 60 test records

### Input Features

| Feature | Type | Description |
|---|---|---|
| `item_category` | Categorical | Tablet, Syrup, Ointment, First Aid, Supplement |
| `days_until_expiry` | Numeric | Days remaining before expiry |
| `quantity_remaining` | Numeric | Current stock quantity |
| `minimum_stock_level` | Numeric | Minimum required stock |
| `is_opened` | Binary (0/1) | Whether the medicine is opened |
| `essential_item` | Binary (0/1) | Whether it is an essential medicine |

### Output Classes

| Class | Meaning |
|---|---|
| 🔴 `Check Expiry` | Medicine is expiring soon or already expired |
| 🟡 `Restock Soon` | Stock is below the minimum level |
| 🟢 `No Immediate Action` | Medicine is in good condition |

### Where the Algorithm is Used in Code

```python
# 1. CREATE — Define the Decision Tree model
model = DecisionTreeClassifier(max_depth=5, random_state=42)

# 2. TRAIN — Fit the model on training data
model.fit(X_train, y_train)

# 3. PREDICT — Classify a new medicine entry
def predict_priority(category, days, qty, min_stock, opened, essential):
    s = pd.DataFrame([{
        "days_until_expiry":   int(days),
        "quantity_remaining":  int(qty),
        "minimum_stock_level": int(min_stock),
        "is_opened":           1 if opened == "Yes" else 0,
        "essential_item":      1 if essential == "Yes" else 0,
        "item_category":       category
    }])
    s = pd.get_dummies(s, columns=["item_category"], dtype=int)
    s = s.reindex(columns=FEATURE_COLS, fill_value=0)
    return model.predict(s)[0]   # ← Decision Tree predicts here
```

---

## 📂 Project Structure

```
flask_app/
│
├── app.py                                      # Main Flask app + ML model training & routes
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation
├── medicine_inventory_expiry_synthetic.csv     # Dataset (300 records)
│
├── templates/                                  # Jinja2 HTML templates
│   ├── base.html                               # Sidebar + topbar layout
│   ├── home.html                               # Landing page with login/signup
│   ├── dashboard.html                          # Main dashboard with charts
│   ├── add_medicine.html                       # Add medicine form
│   ├── predictions.html                        # AI predictions for all medicines
│   ├── inventory.html                          # Full inventory table
│   ├── insights.html                           # Model performance & feature importance
│   └── about.html                              # Project info page
│
└── static/
    ├── css/
    │   └── style.css                           # All styles for home + app layout
    └── js/
        └── main.js                             # Topbar clock updater
```

---

## ▶️ How to Run

### Prerequisites
- Python 3.8 or above
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/meditrack-flask.git

# 2. Navigate into the project folder
cd meditrack-flask

# 3. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the Flask app
python app.py

# 6. Open in your browser
http://127.0.0.1:5000
```

> **Login:** Use any email and password — the app runs in demo mode.

---

## ⚠️ Disclaimer

This is an **educational prototype** built as a Foundations of AI case study.
Predictions made by the model are for **inventory management purposes only** and do **not** constitute medical advice.
Always consult a qualified pharmacist or healthcare professional for medical decisions.

---

*Built with ♥ — Better inventory management for a healthier tomorrow.*
