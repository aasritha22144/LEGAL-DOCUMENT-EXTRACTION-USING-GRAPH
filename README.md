# 🏛️ Legal Document Extraction Using Graph Representation

This project presents a robust **Legal Document Extraction System** that leverages **Natural Language Processing (NLP)** and **Knowledge Graphs** to convert unstructured legal documents into structured, visualized insights. The system includes a PyQt5 GUI for easy interaction, supports NER (Named Entity Recognition), Relation Extraction, and displays results as graphs using NetworkX. Optionally, graphs can be stored in **Neo4j** for advanced querying.

## 📌 Features

- 📂 Load legal documents (PDF, DOCX, TXT)
- 🧠 NLP pipeline using spaCy and Legal-BERT
- 🔍 Named Entity Recognition (NER) for legal-specific entities
- 🔗 Relation Extraction with context-based classification
- 🌐 Knowledge Graph generation using NetworkX
- 🧩 Optional graph storage in Neo4j database
- 🖼️ GUI built with PyQt5 for interactive use
- 📝 Abstractive summarization using T5 model

---

## 🧠 Technologies Used

| Module | Technology |
|--------|------------|
| **GUI** | PyQt5 |
| **NER/RE** | spaCy, Legal-BERT |
| **Summarization** | T5 (Text-to-Text Transfer Transformer) |
| **Graph Visualization** | NetworkX |
| **Database (Optional)** | Neo4j (Cypher queries supported) |
| **Language** | Python 3.8+ |

---

## 🔧 Project Architecture

### 1. **Legal Document Processing Pipeline**
- **Config Module:** Validates dataset paths and types (IN-ABS / IN-EXT)
- **DataLoader:** Converts uploaded documents to plain text
- **spaCy + Rule-based NLP:** Extracts base entities and relations
- **Legal-BERT:** Enhances contextual accuracy
- **Graph Construction:** Builds interpretable knowledge graphs (NetworkX)
- **Neo4j (optional):** Stores graph in database for later querying

### 2. **Advanced NLP Pipeline**
- **LegalExtractionModel (Legal-BERT):**
  - Classifies document sections (Civil/Criminal/etc.)
  - Extracts NER and Relations
- **T5 Summarizer:**
  - Abstractive summarization via encoder-decoder
  - Evaluated with ROUGE metrics

---

## 📊 Sample Output Snapshots

- **📄 Raw Text Ingestion & Metadata Extraction**
- **🔍 Entity Extraction (Judge, Accused, Law, Court, Date...)**
- **🔗 Relation Graphs:**
  - _"Judge presided over Case"_
  - _"Law applied to Accused"_
- **🧭 Visual Graphs:**
  - Nodes = Entities (color-coded)
  - Edges = Relationships (labeled)
- **📈 Summarization Metrics:**
  - ROUGE-1, ROUGE-2, ROUGE-L

---

## 🚀 How to Run

```bash

# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the GUI
python main.py
