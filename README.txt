# 🤖 CEAT-OCS RAG Chatbot

**An AI-Powered Help Desk Assistant for Student Support Services**  
*College of Engineering and Agro-Industrial Technology (CEAT)*  
*Office of the College Secretary*  
*University of the Philippines Los Baños*

---

## 📋 Overview

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** designed to assist CEAT students with common inquiries about policies, procedures, forms, scholarships, and administrative matters. By leveraging Large Language Models (LLMs) and vector-based semantic search, the chatbot provides accurate, grounded responses drawn directly from official CEAT documentation.

**Key Features:**
-  **Grounded Responses:** All answers sourced from official CEAT policy documents
-  **Fast Response Time:** Average 3-5 seconds per query
-  **Cost-Effective:** ~$0.0004 per query (~₱0.02)
-  **Knowledge Base:** Covers enrollment, graduation, scholarships, shifting, forms, and contacts
-  **Semantic Search:** Understands student intent beyond keyword matching
-  **User-Friendly Interface:** Simple Streamlit-based chat UI

---

## 🎯 Project Status

**Current Version:** v1.0 (Proof of Concept)  
**Status:** ✅ Validated & Tested | ⚠️ Not Production-Ready

This system has been comprehensively evaluated through:
- **Functional Testing:** 80% pass rate (4/5 tests)
- **RAGAS Evaluation:** 74% correctness, 68% relevance
- **Load Testing:** 81% success rate with 50 concurrent users
- **User Experience:** 91% automated score, 4.58/5 student satisfaction

**For Production Deployment:** See [Recommendations](#-recommendations-for-production) below.

---

## 🏗️ Architecture
```
┌─────────────────┐
│  Streamlit UI   │ ← Student interaction
└────────┬────────┘
         │
    ┌────▼─────────────────────────────────────┐
    │         RAG Pipeline                      │
    │  ┌──────────┐  ┌──────────┐  ┌─────────┐│
    │  │Retriever │→ │  Prompt  │→ │   LLM   ││
    │  │(Top 10)  │  │ Template │  │GPT-4o-  ││
    │  └──────────┘  └──────────┘  │  mini   ││
    │       ↑                       └─────────┘│
    └───────┼────────────────────────────────┬─┘
            │                                │
    ┌───────▼────────┐              ┌───────▼──────┐
    │  Qdrant Vector │              │  OpenAI API  │
    │     Store      │              │  Embeddings  │
    │  (In-Memory)   │              │  + LLM       │
    └────────────────┘              └──────────────┘
            ↑
    ┌───────┴────────────────────┐
    │  Document Processing       │
    │  ┌──────────────────────┐  │
    │  │ DirectoryLoader      │  │
    │  │ (PyMuPDF)            │  │
    │  └──────┬───────────────┘  │
    │         │                   │
    │  ┌──────▼───────────────┐  │
    │  │ RecursiveTextSplitter│  │
    │  │ (800 chars, 100 overlap)│
    │  └──────┬───────────────┘  │
    │         │                   │
    │  ┌──────▼───────────────┐  │
    │  │ OpenAIEmbeddings     │  │
    │  │ (text-embedding-3-   │  │
    │  │  small, 1536-dim)    │  │
    │  └──────────────────────┘  │
    └────────────────────────────┘
            ↑
    ┌───────┴────────┐
    │  CEAT Policy   │
    │  PDFs (data/)  │
    └────────────────┘
```

**Technology Stack:**
- **LLM:** OpenAI GPT-4o-mini
- **Embeddings:** text-embedding-3-small (1,536 dimensions)
- **Vector Database:** Qdrant (in-memory)
- **Framework:** LangChain
- **Frontend:** Streamlit
- **Language:** Python 3.10+

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- CEAT policy documents (PDFs)

### Installation

1. **Clone the repository:**
```bash
   git clone https://github.com/your-org/ceat-ocs-rag-chatbot.git
   cd ceat-ocs-rag-chatbot
```

2. **Create virtual environment:**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

4. **Configure environment:**
```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key:
   # OPENAI_API_KEY=sk-your-key-here
```

5. **Add CEAT documents:**
```bash
   # Place your CEAT policy PDFs in the data/ folder
   mkdir -p data
   # Copy PDFs to data/
```

6. **Run the chatbot:**
```bash
   streamlit run app.py
```

7. **Access the interface:**
```
   Open browser to: http://localhost:8501
```

---

## 📁 Project Structure
```
ceat-ocs-rag-chatbot/
├── app.py                      # Main Streamlit application
├── test_chatbot.py             # Functional testing suite (pytest)
├── eval_gpt4.py                # GPT-4 LLM-as-judge evaluation
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .env                        # Your API keys (DO NOT COMMIT)
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── data/                       # CEAT policy PDFs (add your documents here)
│   ├── graduation_policy.pdf
│   ├── scholarship_guidelines.pdf
│   └── ...
├── results/                    # Evaluation results (generated)
│   └── ux_evaluation_results.json
└── docs/                       # Additional documentation
    ├── ARCHITECTURE.md
    ├── TESTING.md
    └── DEPLOYMENT.md
```

---

## 🧪 Testing

### Run Functional Tests
```bash
pytest test_chatbot.py -v
```

**Tests include:**
- Document loading
- Text chunking
- Retrieval accuracy
- Response generation
- Hallucination detection

### Run GPT-4 UX Evaluation
```bash
python eval_gpt4.py
```

**Evaluates:**
- Relevance (does it answer the question?)
- Completeness (sufficient detail?)
- Clarity (easy to understand?)
- Actionability (can students act on it?)

---

## 📊 Performance Metrics

| Metric | Result | Notes |
|--------|--------|-------|
| **Functional Tests** | 80% (4/5 passed) | One hallucination test failed |
| **RAGAS Correctness** | 74% | Industry-standard RAG evaluation |
| **RAGAS Relevance** | 68% | Answer-question alignment |
| **UX Quality (Automated)** | 91% | Rule-based scoring |
| **UX Quality (Survey)** | 4.58/5 (91.6%) | Student feedback (n=10) |
| **Load Test Success** | 81% @ 50 users | 19% failed due to rate limits |
| **P95 Latency** | 4.8s | 95% of queries under 5 seconds |
| **Cost per Query** | ~$0.0004 | ~₱0.02 per interaction |

---

## ⚠️ Known Limitations

### Current Version (v1.0)

1. **In-Memory Vector Store**
   - Data lost on restart
   - Not suitable for production
   - **Solution:** Use persistent Qdrant or Pinecone

2. **Static Document Updates**
   - Requires manual PDF replacement
   - No admin interface for updates
   - **Solution:** Build CMS with upload UI

3. **Hallucination Detection**
   - Attempts to answer out-of-domain queries
   - No intent classification pre-filter
   - **Solution:** Add query classification layer

4. **Scalability Constraints**
   - Single instance (no load balancing)
   - OpenAI API rate limits cause 19% failures under load
   - **Solution:** Multi-instance deployment + caching

5. **No Conversation Memory**
   - Each query independent
   - Cannot reference previous messages
   - **Solution:** Implement conversation history (LangChain ConversationBufferMemory)

---

## 🔧 Recommendations for Production

### Priority 1: Infrastructure (0-3 months)

- [ ] **Persistent Vector Database**
  - Replace in-memory Qdrant with persistent storage
  - Consider managed services (Pinecone, Weaviate)
  
- [ ] **API Rate Limit Handling**
  - Upgrade OpenAI API tier
  - Implement request queuing and retries
  - Add response caching for FAQ

- [ ] **Monitoring & Logging**
  - Set up LangSmith or similar for query tracking
  - Monitor costs, latency, errors
  - Alert on anomalies

### Priority 2: Content Management (3-6 months)

- [ ] **Admin UI for Document Management**
  - Upload/update/delete PDFs without coding
  - Automatic re-embedding on changes
  - Version control for documents

- [ ] **Document Quality Improvements**
  - Remove redundancy across policies
  - Standardize formatting
  - Add metadata (effective dates, applicability)

### Priority 3: Enhanced Capabilities (6-12 months)

- [ ] **Hallucination Mitigation**
  - Intent classification pre-filter
  - Confidence scoring
  - Human escalation workflow

- [ ] **Conversation Memory**
  - Multi-turn dialogue support
  - Context retention across questions

- [ ] **Multi-Modal Support**
  - Image upload (e.g., forms, IDs)
  - File attachment responses

- [ ] **Analytics Dashboard**
  - Common queries
  - Unanswered questions
  - User satisfaction trends

---

## 💰 Cost Estimates

### Development/One-Time Costs

- Document embedding (initial): ~$0.004
- Testing (RAGAS evaluation): ~$3.11
- **Total:** ~$3.12

### Operational Costs (Monthly)

| Usage Level | Queries/Month | Cost/Month | Annual Cost |
|-------------|---------------|------------|-------------|
| **Light** | 500 | $0.20 | $2.40 |
| **Moderate** | 2,000 | $0.84 | $10.08 |
| **Heavy** | 10,000 | $4.20 | $50.40 |

**Compare to staff time:**  
- Staff member answering 10,000 emails/month: ~166 hours @ ₱500/hr = **₱83,000 (~$1,500)**
- Chatbot handling same volume: **<$50/month**

**ROI:** 30:1 cost savings

---

## 🤝 Contributing

This project is maintained for **CEAT-UPLB institutional use**. Contributions are welcome from:

- CEAT faculty and staff
- UPLB IT personnel
- Students working on approved projects

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add your feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow existing code style
- Add tests for new features
- Update documentation
- Keep API keys and sensitive data out of commits

---

## 📄 License

**For Internal CEAT-UPLB Use Only**

This software is developed for the College of Engineering and Agro-Industrial Technology (CEAT), University of the Philippines Los Baños. 

**Restrictions:**
- Not licensed for external distribution
- Intended for CEAT administrative and educational use
- Contact CEAT-OCS for permission to adapt or redistribute

**Third-Party Components:**
- LangChain: MIT License
- Streamlit: Apache 2.0 License
- OpenAI API: Subject to OpenAI Terms of Service

---

## 👥 Credits

**Developer:** Alyssa Mila Marie F. Añoso  
**Thesis Adviser:** Engr. JF Chan
**Institution:** College of Engineering and Agro-Industrial Technology (CEAT), UPLB  
**Academic Year:** 2025-2026

**Special Thanks:**
- CEAT Office of the College Secretary staff
- Student testers and survey participants

---

## 📞 Contact & Support

### For CEAT Staff/Faculty

**Technical Issues:**
- GitHub Issues: [github.com/your-org/ceat-ocs-rag-chatbot/issues](https://github.com/your-org/ceat-ocs-rag-chatbot/issues)
- Email: afanoso@up.edu.ph

**Deployment Questions:**
- Contact CEAT-OCS: ceat_ocs@uplb.edu.ph
- Office Hours: Monday-Friday, 8:00 AM - 5:00 PM

### For Future Developers


**Getting Started:**
1. Read this README thoroughly
2. Review the thesis document (if available)
3. Run tests to verify your setup
4. Experiment with test queries
5. Review code comments in `app.py`

---

## 🔮 Future Vision

### Short-Term (6-12 months)
- [ ] Deploy to CEAT-OCS production with admin UI
- [ ] Expand knowledge base with more policy documents
- [ ] Add multilingual support (Filipino/Taglish)

### Medium-Term (1-2 years)
- [ ] Scale to other UPLB colleges (CAS, CDC, CAFS, etc.)
- [ ] Integrate with UPLB CRS (Course Registration System)
- [ ] Mobile app version

### Long-Term (2-5 years)
- [ ] University-wide knowledge management platform
- [ ] Voice interface for accessibility
- [ ] Predictive analytics (anticipate student needs)
- [ ] Integration with UPLB Moodle, email systems

---

## 📚 References & Further Reading

### Documentation

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Qdrant Vector Database](https://qdrant.tech/documentation/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Related Projects

- [LangChain RAG Tutorials](https://python.langchain.com/docs/use_cases/question_answering/)
- [RAGAS Evaluation Framework](https://docs.ragas.io/)
- [Chatbot Arena Leaderboard](https://chat.lmsys.org/)

---

## 🎓 Academic Context

This project was developed as an undergraduate thesis (EE 200) at UPLB, demonstrating:

- **Technical Contribution:** Implementation of RAG architecture for institutional knowledge management
- **Validation:** Comprehensive multi-dimensional evaluation (functional, RAGAS, load, UX)
- **Practical Impact:** Prototype demonstrating 70-80% workload reduction potential for CEAT-OCS
- **Research Value:** Empirical evidence that RAG systems are viable for university administrative support

**Thesis Title:** *"Design and Implementation of an LLM-Powered Help Desk Chatbot with Basic Retrieval-Augmented Generation for Student Support Services at the College of Engineering and Agro-Industrial Technology (CEAT) Office of the College Secretary, University of the Philippines Los Baños"*

---

## ⚡ Quick Links

- 📖 [Full Documentation](docs/)
- 🐛 [Report Issues](https://github.com/your-org/ceat-ocs-rag-chatbot/issues)
- 💬 [Discussions](https://github.com/your-org/ceat-ocs-rag-chatbot/discussions)
- 📊 [Project Board](https://github.com/your-org/ceat-ocs-rag-chatbot/projects)
- 🎯 [Roadmap](https://github.com/your-org/ceat-ocs-rag-chatbot/milestones)

---

## ✨ Version History

### v1.0 (December 2024)
- ✅ Initial proof-of-concept release
- ✅ Core RAG pipeline implementation
- ✅ Streamlit UI
- ✅ Comprehensive testing suite
- ✅ Documentation complete

### v1.1 (Planned - Q1 2025)
- 🔄 Persistent vector storage
- 🔄 Basic admin UI
- 🔄 Improved hallucination detection
- 🔄 Enhanced monitoring

### v2.0 (Planned - Q2 2025)
- 🔄 Production deployment
- 🔄 Full CMS integration
- 🔄 Multi-instance scaling
- 🔄 Advanced analytics

---

<div align="center">

**Built with ❤️ for CEAT-UPLB**

*Empowering student support through AI*

[⬆ Back to Top](#-ceat-ocs-rag-chatbot)

</div>
