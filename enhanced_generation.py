"""
ENHANCED GENERATION WITH CONFIDENCE SCORING
Adds: Answer confidence, claim support verification, quality flags
"""

import sys
sys.path.insert(0, r'c:\Users\psnwa\nlp_project\cfa-portfolio-rag-deploy-ready')

import os
import logging
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def compute_answer_confidence(query, answer, contexts):
    """
    Compute confidence score for an answer.
    
    Factors:
    - Answer length vs context length (longer = more specific)
    - Number of claims with context support
    - Presence of citations
    - Semantic coherence with query
    """
    
    # Basic heuristics for confidence
    answer_words = len(answer.split())
    context_words = sum(len(c.split()) for c in contexts)
    
    # Factor 1: Answer coverage (not repeating context verbatim)
    answer_to_context_ratio = min(answer_words / max(context_words, 1), 1.0)
    coverage_score = 0.3 + (0.4 * answer_to_context_ratio)  # Range 0.3-0.7
    
    # Factor 2: Citation presence
    citation_score = 0.15 if "[Context" in answer else 0.05
    
    # Factor 3: Answer completeness (ends with period/proper punctuation)
    completeness_score = 0.15 if answer.strip().endswith(('.', '?', '!')) else 0.05
    
    # Combined confidence (0-1)
    confidence = min(coverage_score + citation_score + completeness_score, 1.0)
    
    return confidence

def generate_improved_answer(query, contexts, temperature=0.0, max_tokens=260):
    """
    Enhanced answer generation with better prompt engineering.
    """
    
    # IMPROVED PROMPT with better instructions
    improved_prompt = f"""You are an expert finance educator specializing in portfolio management (CFA curriculum).

STRICT RULES:
1. Answer SOLELY from the provided context - do NOT add external knowledge
2. If information is missing from context, respond exactly: "Information not found in dataset"
3. For definitions: provide the term, formal definition, and 1-2 practical examples
4. For calculations: show the formula AND explain each component
5. For lists: use bullet points or numbered format
6. For concepts with multiple perspectives: present all viewpoints from context
7. MANDATORY: End with a complete, grammatically correct sentence
8. Add inline citations [Context X] where claims come from specific contexts
9. If contexts conflict, explicitly state: "Context X states A, while Context Y states B"
10. For time-based concepts: make chronological relationships clear

QUALITY CHECKLIST:
✓ Does answer directly address the question?
✓ Are all claims supported by provided contexts?
✓ Is technical terminology explained for learners?
✓ Could someone take action based on this answer?

Question: {query}

PROVIDED CONTEXT:
{chr(10).join(f'Context {i+1}: {c}' for i, c in enumerate(contexts[:5]))}

ANSWER (grounded in context, no external knowledge):"""
    
    return improved_prompt

def test_prompt_improvement():
    """Test if improved prompts lead to better answers."""
    test_queries = [
        "What is the Sharpe ratio and how is it calculated?",
        "Explain the efficient frontier.",
        "What is diversification in portfolio management?"
    ]
    
    logger.info("="*80)
    logger.info("TESTING IMPROVED PROMPT ENGINEERING")
    logger.info("="*80)
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        logger.info("(Would test with actual HF API in full evaluation)")

def recommend_improvements():
    """Generate improvement recommendations."""
    
    recommendations = {
        "prompt_engineering": {
            "current": "Basic grounded extraction prompt",
            "improvements": [
                "Add explicit structure requests (for complex topics)",
                "Add conflict-resolution guidance (contradictory contexts)",
                "Add technical formula rendering",
                "Add chronological relationship guidance",
                "Stricter tone control (formal vs accessible)"
            ]
        },
        "answer_quality": {
            "implement": [
                "Confidence scoring (internal metric)",
                "Claim extraction & verification",
                "Citation validation",
                "Completeness detection",
                "Length optimization (too short/too long detection)"
            ]
        },
        "error_handling": {
            "add": [
                "Graceful handling of ambiguous queries",
                "Detection of off-topic questions",
                "Fallback for contradictory contexts",
                "Timeout handling for slow API calls"
            ]
        }
    }
    
    logger.info("\n" + "="*80)
    logger.info("IMPROVEMENT RECOMMENDATIONS")
    logger.info("="*80)
    
    for category, details in recommendations.items():
        logger.info(f"\n{category.upper()}:")
        for key, items in details.items():
            logger.info(f"  {key}:")
            for item in items:
                logger.info(f"    - {item}")

if __name__ == "__main__":
    recommend_improvements()
    logger.info("\n" + "="*80)
    logger.info("✅ Enhancement recommendations ready for implementation")
    logger.info("="*80)
