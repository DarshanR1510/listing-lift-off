# 🚀 Listing Lift-Off Strategy

A powerful scanner that identifies **IPO stocks** showing fresh momentum right after listing, based on **ATH breakout continuation strategy**. Designed to detect early trend opportunities in newly listed companies.

## 🎯 Strategy Logic

This strategy focuses on **IPO listings within last 1 year** and captures breakout continuation moves after the stock breaks its **All-Time High (ATH)**.

### **Entry Criteria**
1. Stock must be an IPO listed within **last 12 months**  
2. **Day 1** – Stock **breaks ATH** and **closes above it**  
3. **Day 2** – **Buy at market open**, unconditionally  

📌 When criteria are met → **Signal Generated = Entry Opportunity Available**

### **Exit Rule**
- Exit when price **closes below EMA 20**

### **Stop Loss**
- **Day 1 candle low** (ATH breakout day)

This setup aims to catch **fresh post-IPO rallies** where a breakout leads to strong upside momentum.

---

## ✨ Features

- 🔍 **Automatic IPO Filtering**
- 📈 **ATH Breakout Detection**
- 📅 **Next-Day Buy Signals**
- 🧠 **Rule-Based System**
- 📊 **Dashboard View**
- 📥 **CSV Export**
- 🌐 **Responsive UI**

---

## 🚀 Quick Start

### Requirements
- Python 3.8+
- Streamlit

### Installation

\`\`\`bash
git clone https://github.com/DarshanR1510/ipo-breakout-scout.git
cd ipo-breakout-scout
pip install -r requirements.txt
streamlit run streamlit_app.py
\`\`\`

---

## 🧠 How It Works

1. Fetch IPO list (last 1-year)
2. Detect **Day-1 ATH Breakout Close**
3. Generate **Day-2 Buy Signal**
4. Manage trade using **EMA20 exit rule**
5. Track results in UI

---

## 📊 Output Columns

| Column | Meaning |
|--------|---------|
| Symbol | Stock ticker |
| IPO Listing Date | First day listed |
| ATH Breakout Date | Breakout candle day |
| Buy Date | Day-2 entry |
| Buy Price | Day-2 open |
| Stop Loss | Day-1 low |
| Exit Date | Close below EMA 20 |
| Gain % | Result |
| Days Held | Duration |

---

## Configurable settings

\`\`\`python
IPO_LOOKBACK_DAYS = 365
EMA_EXIT = 20
\`\`\`

---

## Disclaimer
Educational purpose only. Not financial advice.

---
