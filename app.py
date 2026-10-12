from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
import json, os, shutil

app = Flask(__name__)
app.secret_key = "meditrack_secret_2024"

# ── Copy CSV to flask_app folder ──────────────────────────────────────────────
BASE = os.path.dirname(__file__)
CSV_SRC  = os.path.join(BASE, "..", "medicine_inventory_expiry_synthetic.csv")
CSV_PATH = os.path.join(BASE, "medicine_inventory_expiry_synthetic.csv")
if not os.path.exists(CSV_PATH):
    shutil.copy(CSV_SRC, CSV_PATH)

# ── Train model once at startup ───────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
X  = df.drop(columns=["item_id", "inventory_priority"])
y  = df["inventory_priority"]
X  = pd.get_dummies(X, columns=["item_category"], dtype=int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ALGORITHM - Create the model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
# ALGORITHM - Train the model
model.fit(X_train, y_train)
preds      = model.predict(X_test)
FEATURE_COLS = list(X.columns)

# ── Model metrics ─────────────────────────────────────────────────────────────
METRICS = {
    "accuracy":  round(accuracy_score(y_test, preds) * 100, 1),
    "precision": round(precision_score(y_test, preds, average="weighted", zero_division=0) * 100, 1),
    "recall":    round(recall_score(y_test, preds, average="weighted", zero_division=0) * 100, 1),
    "f1":        round(f1_score(y_test, preds, average="weighted", zero_division=0) * 100, 1),
}
cm     = confusion_matrix(y_test, preds, labels=["Check Expiry","Restock Soon","No Immediate Action"])
METRICS["cm"] = cm.tolist()

# ── Medicine name pools per category ─────────────────────────────────────────
MED_NAMES = {
    "Tablet":     ["Paracetamol","Ibuprofen","Amoxicillin","Metformin","Atorvastatin",
                   "Cetirizine","Azithromycin","Omeprazole","Aspirin","Ciprofloxacin",
                   "Doxycycline","Losartan","Amlodipine","Pantoprazole","Montelukast"],
    "Syrup":      ["Benadryl","Cough Syrup","Amoxicillin Syrup","Paracetamol Syrup",
                   "Ibuprofen Syrup","Cetirizine Syrup","Antacid Syrup","Iron Tonic",
                   "Vitamin C Syrup","Zinc Syrup"],
    "Ointment":   ["Betadine","Neosporin","Clotrimazole","Hydrocortisone","Mupirocin",
                   "Diclofenac Gel","Calamine Lotion","Ketoconazole Cream","Zinc Oxide","Acyclovir"],
    "First Aid":  ["Bandage Roll","Antiseptic Wipes","Cotton Gauze","Adhesive Plaster",
                   "Elastic Bandage","Sterile Dressing","Medical Tape","Wound Closure",
                   "Triangular Bandage","Eye Pad"],
    "Supplement": ["Vitamin D3","Vitamin B12","Omega-3","Calcium Carbonate","Iron Supplement",
                   "Zinc Tablet","Magnesium","Multivitamin","Folic Acid","Biotin"],
}
import random as _random
_random.seed(42)

def _assign_name(row):
    pool = MED_NAMES.get(row["item_category"], [row["item_category"]])
    idx  = int(row["item_id"].replace("MED", "")) - 1
    return pool[idx % len(pool)]

# ── In-memory inventory ───────────────────────────────────────────────────────
inventory = df.to_dict("records")
for _r in inventory:
    _r["med_name"] = _assign_name(_r)

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
    
    # ALGORITHM - Use the model to predict
    return model.predict(s)[0]

def get_stats():
    n_exp = sum(1 for r in inventory if r["inventory_priority"] == "Check Expiry")
    n_rst = sum(1 for r in inventory if r["inventory_priority"] == "Restock Soon")
    n_ok  = sum(1 for r in inventory if r["inventory_priority"] == "No Immediate Action")
    return len(inventory), n_exp, n_rst, n_ok

def get_category_counts():
    counts = {}
    for r in inventory:
        c = r["item_category"]
        counts[c] = counts.get(c, 0) + 1
    return counts

# ── Routes ────────────────────────────────────────────────────────────────────
# simple in-memory user store for demo
USERS = {}  # {email: password}

@app.route("/")
def home():
    error   = request.args.get("error", "")
    success = request.args.get("success", "")
    mode    = request.args.get("mode", "login")   # login | signup
    return render_template("home.html", error=error, success=success, mode=mode)

@app.route("/login", methods=["POST"])
def login():
    email    = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()
    if not email:
        return redirect(url_for("home", error="Please enter your email address.", mode="login"))
    # accept any credentials — demo mode
    session["user"] = email.split("@")[0] if "@" in email else email
    return redirect(url_for("dashboard"))

@app.route("/signup", methods=["POST"])
def signup():
    email    = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()
    if not email:
        return redirect(url_for("home", error="Please enter your email address.", mode="signup"))
    if not password:
        return redirect(url_for("home", error="Please enter a password.", mode="signup"))
    # auto-login after signup
    session["user"] = email.split("@")[0] if "@" in email else email
    return redirect(url_for("dashboard"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("home"))
    total, n_exp, n_rst, n_ok = get_stats()
    cat_counts = get_category_counts()
    recent = inventory[:6]
    return render_template("dashboard.html",
        user=session["user"], total=total, n_exp=n_exp,
        n_rst=n_rst, n_ok=n_ok, cat_counts=cat_counts,
        recent=recent, metrics=METRICS)

@app.route("/add", methods=["GET","POST"])
def add_medicine():
    if "user" not in session:
        return redirect(url_for("home"))
    success = request.args.get("success", "")
    if request.method == "POST":
        form_data = request.form.to_dict()
        priority = predict_priority(
            form_data["category"], form_data["days"],
            form_data["qty"], form_data["min_stock"],
            form_data["opened"], form_data["essential"])
        new_id = f"MED{len(inventory)+1:03d}"
        new_row = {
            "item_id": new_id,
            "med_name": form_data.get("med_name", ""),
            "item_category": form_data["category"],
            "days_until_expiry": int(form_data["days"]),
            "quantity_remaining": int(form_data["qty"]),
            "minimum_stock_level": int(form_data["min_stock"]),
            "is_opened": 1 if form_data["opened"] == "Yes" else 0,
            "essential_item": 1 if form_data["essential"] == "Yes" else 0,
            "inventory_priority": priority
        }
        inventory.insert(0, new_row)
        name = form_data.get("med_name") or new_id
        return redirect(url_for("add_medicine", success=f"{name} added successfully with priority: {priority}"))
    return render_template("add_medicine.html", user=session["user"], success=success)

@app.route("/predictions/refresh")
def refresh_predictions():
    if "user" not in session:
        return redirect(url_for("home"))
    # Re-run predict_priority on every inventory item and update stored priority
    for r in inventory:
        r["inventory_priority"] = predict_priority(
            r["item_category"],
            r["days_until_expiry"],
            r["quantity_remaining"],
            r["minimum_stock_level"],
            "Yes" if r["is_opened"] else "No",
            "Yes" if r["essential_item"] else "No"
        )
    return redirect(url_for("predictions", refreshed=1))

@app.route("/predictions")
def predictions():
    if "user" not in session:
        return redirect(url_for("home"))
    refreshed = request.args.get("refreshed", 0, type=int)
    # Run prediction on every item in inventory
    results = []
    for r in inventory:
        cat = r["item_category"]
        days = r["days_until_expiry"]
        qty  = r["quantity_remaining"]
        mn   = r["minimum_stock_level"]
        opened    = "Yes" if r["is_opened"] else "No"
        essential = "Yes" if r["essential_item"] else "No"
        priority  = predict_priority(cat, days, qty, mn, opened, essential)
        results.append({
            "item_id":   r["item_id"],
            "med_name":  r.get("med_name") or cat,
            "category":  cat,
            "days":      days,
            "qty":       qty,
            "min_stock": mn,
            "opened":    opened,
            "essential": essential,
            "priority":  priority,
        })
    # summary counts
    n_exp = sum(1 for r in results if r["priority"] == "Check Expiry")
    n_rst = sum(1 for r in results if r["priority"] == "Restock Soon")
    n_ok  = sum(1 for r in results if r["priority"] == "No Immediate Action")
    return render_template("predictions.html",
                           user=session["user"], results=results,
                           n_exp=n_exp, n_rst=n_rst, n_ok=n_ok,
                           refreshed=refreshed)


@app.route("/inventory")
def view_inventory():
    if "user" not in session:
        return redirect(url_for("home"))
    search   = request.args.get("search", "").lower()
    filter_p = request.args.get("priority", "All")
    view = inventory
    if search:
        view = [r for r in view if search in r["item_id"].lower()
                or search in r["item_category"].lower()
                or search in r.get("med_name","").lower()]
    if filter_p != "All":
        view = [r for r in view if r["inventory_priority"] == filter_p]
    return render_template("inventory.html",
                           user=session["user"], inventory=view,
                           search=search, filter_p=filter_p)

@app.route("/insights")
def model_insights():
    if "user" not in session:
        return redirect(url_for("home"))
    imp = sorted(zip(FEATURE_COLS, model.feature_importances_),
                 key=lambda x: x[1], reverse=True)[:8]
    return render_template("insights.html",
                           user=session["user"], metrics=METRICS,
                           feature_imp=imp)

@app.route("/about")
def about():
    if "user" not in session:
        return redirect(url_for("home"))
    return render_template("about.html", user=session["user"])

if __name__ == "__main__":
    app.run(debug=True, port=5000)
