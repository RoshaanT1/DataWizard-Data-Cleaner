# 📊 AI-Powered Data Exploration Platform

Welcome to our submission for the **University Hackathon** hosted by the **Data Insights Club**! This platform empowers **non-technical users** to derive deep, actionable insights from real-world datasets — without writing a single line of code.

---

## 🎯 Objective

Build an intuitive, AI-driven data exploration tool that:
- Uploads and processes messy CSV/Excel data
- Cleans and profiles the dataset using AI
- Generates smart visualizations and natural language summaries
- Allows users to query data in plain English
- Offers autonomous insights via an AI "Insight Agent"
- Exports polished, shareable reports

---

## 🧠 Core Features

### 📂 Upload Datasets
- Supports CSV and Excel formats
- Drag-and-drop interface or file picker

### 🧼 Auto Data Cleaning & Profiling
- AI detects column types, missing values, and outliers
- Automatically applies fixes (e.g., missing value imputation)
- Provides a profile summary with distributions and basic stats

### 📊 Smart Visualizations & Narrative Summaries
- Auto-generated charts (bar, line, pie, etc.)
- Descriptive narratives for each visualization
- Dynamic chart selection based on data type

### 💬 Natural Language Querying
- Users ask questions like:  
  “What are the top complaints by year?”  
  “Show average satisfaction by age group.”
- AI generates a visual + text-based response

### 🧠 Insight Agent (Agentic AI Mode)
- One-click autonomous analysis
- Profiles the dataset
- Detects hidden trends, anomalies, and clusters
- Suggests visualizations, filters, and next questions
- Responds interactively:  
  _“Would you like to compare by location or product category?”_
- Outputs an actionable visual + narrative summary

### 📤 Exportable Reports
- Create PDF or HTML reports with:
  - Key stats
  - Charts
  - AI-generated narratives
- Perfect for sharing or presentations

---

## 🧪 Sample Use Cases

- **Public Opinion Data**: Analyze satisfaction, trust, and media perception
- **Hotel Reviews**: Uncover service weak points and customer sentiment
- **E-Commerce Complaints**: Detect product/delivery issues
- **Education Feedback**: Visualize trends and predict dropouts

---

## ✨ Bonus Features (Optional Add-ons)

| Feature | Description | Status |
|--------|-------------|--------|
| 🎙️ Voice-to-Text | Users speak queries like “Show sales by quarter” | ✅ Implemented |
| 📖 Pattern Explanation | Insight Agent explains *why* a trend matters | ✅ Implemented |
| 🧱 Custom Dashboards | Drag-and-drop visual layout builder | ✅ Implemented |
| 📣 Social-Style Posts | Auto-generates LinkedIn-style insights | 🔄 In Progress |

---

## 📂 Sample Dataset

- File: `Ecommerce Sales Analysis.xlsx`
- Contains: User feedback, timestamps, issue categories, resolutions

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+

### Setup

```bash
git clone https://github.com/RoshaanT1/DataWizard-Data-Cleaner.git
cd DataWizard-Data-Cleaner
```
Python (Backend)
```bash
pip install -r requirements.txt
```

To run it:

```bash
python app.py
```
## 🧰 Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript (Vanilla JS), Bootstrap 5, Bootstrap Icons
- **Backend**: Python, Flask
- **AI/LLM**: Hugging Face Transformers/ Deepseek
- **Data Processing**: Pandas, NumPy, Pandas Profiling, Scikit-learn
- **Visualization**: Plotly, Matplotlib, Seaborn, Plotly.js
- **NLP & Agents**: LangChain
- **Reporting**: WeasyPrint / pdfkit / HTML templates

---


**Examples:**
- Dataset upload
- Natural language query response
- Insight Agent recommendations
- Exported report preview

---

## 👥 Team Members

|      Name     |              Role                    |
|---------------|--------------------------------------|
| Roshaan Tahir |  Developer & Agentic AI              |
| M.Hammad Khan |  Backend Developer & Speech to Text  |

---

## 📦 Deliverables

- ✅ Working prototype  
- ✅ Source code  
- ✅ Sample dataset    
- ✅ README

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgments

Thanks to the **Data Insights Club** and our university for organizing this hackathon!
