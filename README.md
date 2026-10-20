# kazakh-russian-nlp
A cascaded NLP system for language identification, code-switching detection, and sentiment analysis of Kazakh–Russian text.

## Project Overview
This project focuses on the development and evaluation of a cascaded Natural Language Processing (NLP) system designed to process texts containing Kazakh, Russian, and code-switched combinations of both languages. In bilingual regions such as Kazakhstan, speakers frequently alternate between Kazakh and Russian within a single sentence or text stream. This linguistic phenomenon—known as code-switching—poses significant challenges for standard monolingual or independent NLP pipelines because language boundaries cannot always be reliably determined at the sentence level, and both languages share the Cyrillic script.

The proposed solution integrates language identification, token-level code-switching detection, and sentiment analysis into a unified, multi-stage cascaded processing pipeline where earlier predictions inform subsequent analysis.

---

## Problem Statement, Target Users, and Expected Outcomes

### Problem Statement
Processing mixed-language texts requires handling multiple interconnected NLP tasks simultaneously. Traditional systems treat language identification, code-switching, and sentiment analysis as isolated tasks. However, errors or misclassifications in language identification or boundary detection propagate downstream and distort sentiment classification. This project investigates how to effectively design, implement, and evaluate a cascaded NLP system to address these challenges using multilingual transformer models.

### Target Users
* **Primary Target Users:**
  * **Bilingual Citizens:** Individuals who naturally alternate between Kazakh and Russian in daily digital communication.
  * **Language Learners:** Students learning either Kazakh or Russian who occasionally mix vocabularies or terms from their stronger language.
* **Secondary Target Users:**
  * **Businesses and Customer Support Providers:** Organizations processing customer feedback, inquiries, or chat logs containing mixed-language text.
  * **Conversational AI & EdTech Developers:** Teams building chatbots, virtual assistants, or digital learning platforms that require robust multilingual and code-switched text analysis modules.

### Expected Outcomes
* Curated and preprocessed datasets tailored for language identification, code-switching detection, and sentiment analysis of Kazakh-Russian texts.
* Fine-tuned multilingual transformer models (leveraging XLM-RoBERTa baselines).
* An integrated, modular cascaded NLP pipeline exposed via a REST API using FastAPI.
* Comprehensive evaluation reports detailing model performance metrics (accuracy, precision, recall, macro-F1, and confusion matrices) alongside an analysis of error propagation across pipeline stages.

---

## Team Members and Supervisor
* **Developer / Student:** Ramazan Kozhabek (Individual Project)
* **Supervisor:** Shynar Akhmetzhanova

---

## Project Scope and Current Development Status

### Project Scope
* **In-Scope:** Data collection and preparation, model fine-tuning for language identification, token-level code-switching detection, sentiment analysis, pipeline integration, REST API development using FastAPI, and systematic evaluation against baseline models (such as TF-IDF with Logistic Regression and fastText).
* **Out-of-Scope:** Real-time speech-to-text audio processing, full-scale production web application deployment, and support for languages other than Kazakh and Russian.

### Current Development Status
* Completed initial project proposal, background research, user interviews, literature review, and existing system/technology comparisons.
* Established requirements specifications, system architecture design, data models, and verification plans.
* Setting up the repository structure, baseline scripts, and environment configurations.

---

## Selected Technologies
* **Programming Language:** Python
* **Deep Learning & NLP Frameworks:** PyTorch, Hugging Face Transformers
* **Core Models:** XLM-RoBERTa (Primary multilingual transformer), fastText / Logistic Regression with TF-IDF (Baselines)
* **Data Processing & Analysis:** pandas, scikit-learn
* **API Framework:** FastAPI
* **Data Storage:** CSV files for initial research datasets / SQLite or PostgreSQL for relational persistence if required

---

## Setup, Execution, and Testing Instructions

### Prerequisites
* Python 3.10 or higher installed.
* Git installed on your local machine.


### Installation & Environment Setup
1. Clone the repository:
   git clone https://github.com/Ramazan-droid/kazakh-russian-nlp.git
   cd kazakh-russian-nlp

2. Create and activate a virtual environment:
   * On Windows:
     python -m venv venv
     venv\Scripts\activate
   * On Linux / macOS:
     python3 -m venv venv
     source venv/bin/activate

3. Install required dependencies:
   pip install --upgrade pip
   pip install -r requirements.txt

### Execution
* Run the FastAPI server (once API scripts are committed):
  uvicorn app.main:app --reload

### Testing
* Run automated tests using pytest:
  pytest
