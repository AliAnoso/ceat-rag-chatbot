"""
RAGAS Evaluation Module for CEAT RAG Chatbot
This module provides evaluation capabilities using RAGAS metrics
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
import pandas as pd
from typing import List, Dict

class RAGASEvaluator:
    """Evaluator for RAG system using RAGAS metrics"""
    
    def __init__(self, rag_chain, retriever):
        """
        Initialize evaluator
        
        Args:
            rag_chain: The RAG chain to evaluate
            retriever: Document retriever
        """
        self.rag_chain = rag_chain
        self.retriever = retriever
        
    def create_test_dataset(self, test_questions: List[Dict]) -> Dataset:
        """
        Create a dataset for RAGAS evaluation
        
        Args:
            test_questions: List of dicts with 'question' and 'ground_truth' keys
            
        Returns:
            Dataset object for RAGAS evaluation
        """
        data = {
            'question': [],
            'answer': [],
            'contexts': [],
            'ground_truth': []
        }
        
        for item in test_questions:
            question = item['question']
            ground_truth = item.get('ground_truth', '')
            
            # Get answer from RAG chain
            answer = self.rag_chain.invoke({"question": question})
            
            # Get contexts
            docs = self.retriever.get_relevant_documents(question)
            contexts = [doc.page_content for doc in docs]
            
            data['question'].append(question)
            data['answer'].append(answer)
            data['contexts'].append(contexts)
            data['ground_truth'].append(ground_truth)
        
        return Dataset.from_dict(data)
    
    def evaluate_system(self, test_dataset: Dataset) -> Dict:
        """
        Evaluate RAG system using RAGAS metrics
        
        Args:
            test_dataset: Dataset to evaluate on
            
        Returns:
            Dictionary with evaluation scores
        """
        result = evaluate(
            test_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )
        
        return result
    
    def generate_evaluation_report(self, results: Dict) -> pd.DataFrame:
        """
        Generate a detailed evaluation report
        
        Args:
            results: Evaluation results from RAGAS
            
        Returns:
            DataFrame with detailed metrics
        """
        df = pd.DataFrame(results)
        return df


# Sample test questions for CEAT domain
CEAT_TEST_QUESTIONS = [
    {
        "question": "How do I apply for graduation?",
        "ground_truth": "To apply for graduation: 1) Fill out the Application for Graduation form, 2) Rename the file as Degree_LastName, FirstName, 3) Submit via CEAT Students Forms Submissions. Non-free tuition students must pay the graduation fee of Php 300."
    },
    {
        "question": "What are the requirements for shifting to another program?",
        "ground_truth": "Students may shift to other degree programs within UPLB after one year of residency. Students from BS ABE, BS EE, and BS IE must undergo an interview. Waitlisted applicants are not allowed to shift. A formal letter to the Dean is required, noted by parents/guardians and recommended by adviser and department chair."
    },
    {
        "question": "Who is the College Secretary?",
        "ground_truth": "The College Secretary is Assoc. Prof. Butch G. Bataller, PhD. He is an Associate Professor in the Department of Chemical Engineering who earned his doctorate from Texas A&M University and master's degree from UP Los Baños."
    },
    {
        "question": "What is the contact information for CEAT OCS?",
        "ground_truth": "CEAT Office of College Secretary contact: Phone: 0998-556-6874, Email: ceat_ocs.uplb@up.edu.ph, Office Hours: Monday to Friday, 8:00 AM - 5:00 PM, Location: Dr. Dante B. De Padua Hall, Pili Drive, UPLB"
    },
    {
        "question": "How do I apply for Leave of Absence?",
        "ground_truth": "To apply for LOA: 1) Write a letter to the Dean stating your reason, 2) Have it signed by parents/guardian and registration adviser, 3) Email to metonio@up.edu.ph, 4) Fill out LOA form once approved, 5) Pay LOA fee of Php 150. Maximum LOA is 1 year, renewable once (not exceeding 2 years total)."
    },
    {
        "question": "What scholarships are available for CEAT students?",
        "ground_truth": "CEAT offers several scholarships: 1) CEAT AA Student Loan Program, 2) CEAT AA Study Now Pay Later, 3) CEAT AA Undergraduate Thesis Grant, 4) CEAT AA Undergraduate Internship Grant, 5) CEAT Adopt-a-Student Program. All applications should be submitted as one PDF file via CEAT Students Forms Submissions."
    },
    {
        "question": "What is the process for dropping courses?",
        "ground_truth": "To drop courses: 1) Make a letter of intent addressed to the Dean (Prof. Rex B. Demafelis), 2) Indicate reason for dropping and seek recommendation from adviser and dept chair, 3) If dropping due to health, attach medical certificate, 4) Submit via CEAT Students Forms Submissions, 5) Payment is 10php per unit. OCS will inform via UP mail if approved/disapproved."
    },
    {
        "question": "How do I get a waiver of prerequisite?",
        "ground_truth": "For waiver of prerequisite: 1) Download the appropriate form for your department from CEAT website, 2) Fill out personal information and course details, 3) Have it signed by the instructor, 4) Attach SPMF form in one PDF, 5) Submit to the appropriate OCS staff based on your program. The waiver must be for courses with instructor approval and valid academic reasons."
    },
    {
        "question": "What is the maximum residence rule for CEAT students?",
        "ground_truth": "Students must finish their program within 1.5× the normal length. For Engineering programs (normally 4 years), students must complete within 6 years. Otherwise, further registration in the college is not allowed."
    },
    {
        "question": "What are the requirements for graduation with honors?",
        "ground_truth": "To graduate with honors: 1) Complete at least 75% of total academic units in residence at UP, 2) Carry at least 15 units (or normal load) each semester, 3) Achieve the prescribed minimum weighted average, 4) Only resident credits are included in final average. Exceptions for underload require documentation for health, course unavailability, or employment."
    }
]


def run_evaluation(rag_chain, retriever):
    """
    Run complete RAGAS evaluation
    
    Args:
        rag_chain: The RAG chain to evaluate
        retriever: Document retriever
        
    Returns:
        Evaluation results and report
    """
    evaluator = RAGASEvaluator(rag_chain, retriever)
    
    print("Creating test dataset...")
    test_dataset = evaluator.create_test_dataset(CEAT_TEST_QUESTIONS)
    
    print("Running RAGAS evaluation...")
    results = evaluator.evaluate_system(test_dataset)
    
    print("\nEvaluation Results:")
    print(f"Faithfulness: {results['faithfulness']:.3f}")
    print(f"Answer Relevancy: {results['answer_relevancy']:.3f}")
    print(f"Context Precision: {results['context_precision']:.3f}")
    print(f"Context Recall: {results['context_recall']:.3f}")
    
    report = evaluator.generate_evaluation_report(results)
    
    return results, report


if __name__ == "__main__":
    # This would be run separately with your initialized RAG system
    print("RAGAS Evaluation Module for CEAT RAG Chatbot")
    print("Import this module and call run_evaluation() with your RAG chain and retriever")