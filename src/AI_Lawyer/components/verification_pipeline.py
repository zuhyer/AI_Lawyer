"""
Verification Pipeline - Validates citations and computes confidence scores.
Ensures answer credibility through citation validation and text overlap analysis.
"""

import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from langchain_core.documents import Document

from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.utils.logging_setup import logger


@dataclass
class VerificationResult:
    """Result of answer verification."""
    is_verified: bool
    confidence_score: float
    cited_sections: List[str]
    citation_match_count: int
    similarity_scores: List[float]
    fallback_message: Optional[str] = None
    details: Dict = None


class VerificationPipeline:
    """
    Verification pipeline for LLM-generated answers.
    Validates citations, computes confidence scores, and ensures answer credibility.
    """
    
    # Regex patterns for different citation types
    CITATION_PATTERNS = {
        'section': r'(?:Section|Sec\.?|§|S\.)\s*(\d+(?:[A-Z])?)',
        'article': r'(?:Article|Art\.?)\s*(\d+(?:[A-Z])?)',
        'rule': r'(?:Rule|R\.?)\s*(\d+(?:[A-Z])?)',
        'schedule': r'(?:Schedule|Sch\.?)\s*(\d+)',
        'case_law': r'(?:Case Law|Judgment|Ruling):\s*([^,\n]+)',
        'act': r'(?:(?:Indian|The)\s+)?(\w+(?:\s+\w+)?)\s+(?:Act|Act,|Code)',
    }
    
    def __init__(self, config_manager: ConfigurationManager = None):
        """
        Initialize VerificationPipeline.
        
        Args:
            config_manager: ConfigurationManager instance
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.verification_config = self.config_manager.get_verification_config()
        self.min_confidence = self.verification_config.min_confidence_threshold
        
        logger.info(f"✅ VerificationPipeline initialized (min_confidence={self.min_confidence})")
    
    def extract_citations(self, text: str) -> List[str]:
        """
        Extract all citations from text.
        
        Args:
            text: Text to extract citations from
            
        Returns:
            List of extracted citations
        """
        citations = []
        
        for citation_type, pattern in self.CITATION_PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                citation = match.group(0)
                citations.append(citation)
                logger.debug(f"Found {citation_type}: {citation}")
        
        # Remove duplicates while preserving order
        unique_citations = []
        seen = set()
        for citation in citations:
            if citation not in seen:
                unique_citations.append(citation)
                seen.add(citation)
        
        logger.info(f"Extracted {len(unique_citations)} unique citations from answer")
        return unique_citations
    
    def validate_citations(self, citations: List[str], 
                          retrieved_chunks: List[Document]) -> Tuple[List[str], float]:
        """
        Validate that extracted citations exist in retrieved chunks.
        
        Args:
            citations: List of citations to validate
            retrieved_chunks: Retrieved document chunks
            
        Returns:
            Tuple of (valid_citations, citation_match_score)
        """
        valid_citations = []
        match_count = 0
        
        # Combine all chunk text
        combined_text = ' '.join([doc.page_content for doc in retrieved_chunks])
        combined_text_lower = combined_text.lower()
        
        for citation in citations:
            citation_lower = citation.lower()
            
            # Check if citation text appears in chunks
            if citation_lower in combined_text_lower:
                valid_citations.append(citation)
                match_count += 1
            
            # Also check for partial matches (e.g., "Section 103" in "Section 103 BNS")
            elif any(part in combined_text_lower for part in citation_lower.split()):
                valid_citations.append(citation)
                match_count += 1
        
        # Citation match score (0-1)
        citation_score = match_count / len(citations) if citations else 0.0
        
        logger.info(f"Citation validation: {match_count}/{len(citations)} citations matched "
                   f"(score={citation_score:.2f})")
        
        return valid_citations, citation_score
    
    def compute_similarity_scores(self, answer: str, 
                                 retrieved_chunks: List[Document]) -> Tuple[List[float], float]:
        """
        Compute text similarity between answer and retrieved chunks.
        
        Args:
            answer: LLM-generated answer
            retrieved_chunks: Retrieved document chunks
            
        Returns:
            Tuple of (individual_scores, average_score)
        """
        from difflib import SequenceMatcher
        
        similarity_scores = []
        
        # Normalize text for comparison
        answer_lower = answer.lower()
        
        for chunk in retrieved_chunks:
            chunk_text_lower = chunk.page_content.lower()
            
            # Use SequenceMatcher for similarity
            matcher = SequenceMatcher(None, answer_lower, chunk_text_lower)
            similarity = matcher.ratio()
            similarity_scores.append(similarity)
        
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
        
        logger.info(f"Text similarity computed: avg={avg_similarity:.2f}, "
                   f"min={min(similarity_scores) if similarity_scores else 0:.2f}, "
                   f"max={max(similarity_scores) if similarity_scores else 0:.2f}")
        
        return similarity_scores, avg_similarity
    
    def compute_overlap_score(self, answer: str, 
                             retrieved_chunks: List[Document]) -> float:
        """
        Compute text overlap score between answer and retrieved chunks.
        
        Args:
            answer: LLM-generated answer
            retrieved_chunks: Retrieved document chunks
            
        Returns:
            Overlap score (0-1)
        """
        # Extract key terms from answer (words > 3 characters)
        answer_terms = set(
            word.lower() for word in answer.split() 
            if len(word) > 3 and not word.startswith('the')
        )
        
        if not answer_terms:
            return 0.5
        
        # Count matching terms in chunks
        matching_terms = set()
        for chunk in retrieved_chunks:
            chunk_terms = set(
                word.lower() for word in chunk.page_content.split() 
                if len(word) > 3 and not word.startswith('the')
            )
            matching_terms.update(answer_terms & chunk_terms)
        
        # Overlap score
        overlap_score = len(matching_terms) / len(answer_terms) if answer_terms else 0.0
        
        logger.info(f"Text overlap computed: {len(matching_terms)}/{len(answer_terms)} "
                   f"terms matched (score={overlap_score:.2f})")
        
        return overlap_score
    
    def compute_confidence_score(self, 
                                avg_similarity: float,
                                citation_score: float,
                                overlap_score: float) -> float:
        """
        Compute final confidence score using weighted components.
        
        Formula:
            confidence = 0.4 * avg_similarity + 0.3 * citation_score + 0.3 * overlap_score
        
        Args:
            avg_similarity: Average similarity score (0-1)
            citation_score: Citation validation score (0-1)
            overlap_score: Text overlap score (0-1)
            
        Returns:
            Final confidence score (0-1)
        """
        confidence = (
            0.4 * avg_similarity +
            0.3 * citation_score +
            0.3 * overlap_score
        )
        
        # Clip to [0, 1]
        confidence = min(1.0, max(0.0, confidence))
        
        logger.info(f"Confidence score computed: "
                   f"0.4*{avg_similarity:.2f} + 0.3*{citation_score:.2f} + 0.3*{overlap_score:.2f} = {confidence:.2f}")
        
        return confidence
    
    def get_fallback_message(self) -> str:
        """Get standard fallback message for low confidence answers."""
        return (
            "⚠️ Insufficient verified legal information available. "
            "Please consult a licensed advocate for accurate legal advice."
        )
    
    def verify_answer(self, answer: str, 
                     retrieved_chunks: List[Document],
                     domain: str = None) -> VerificationResult:
        """
        Verify LLM-generated answer against retrieved chunks.
        
        Args:
            answer: LLM-generated answer
            retrieved_chunks: Retrieved document chunks
            domain: Domain used for retrieval (optional)
            
        Returns:
            VerificationResult with verification details
        """
        try:
            logger.info(f"🔍 Starting answer verification (domain={domain})")
            
            # Step 1: Extract citations from answer
            citations = self.extract_citations(answer)
            
            # Step 2: Validate citations against retrieved chunks
            valid_citations, citation_score = self.validate_citations(citations, retrieved_chunks)
            
            # Step 3: Compute text similarity
            similarity_scores, avg_similarity = self.compute_similarity_scores(answer, retrieved_chunks)
            
            # Step 4: Compute text overlap
            overlap_score = self.compute_overlap_score(answer, retrieved_chunks)
            
            # Step 5: Compute final confidence score
            confidence_score = self.compute_confidence_score(
                avg_similarity, citation_score, overlap_score
            )
            
            # Step 6: Determine if answer meets confidence threshold
            is_verified = confidence_score >= self.min_confidence
            
            # Step 7: Prepare fallback message if needed
            fallback_message = None
            if not is_verified:
                fallback_message = self.get_fallback_message()
                logger.warning(f"⚠️  Answer confidence ({confidence_score:.2f}) below threshold ({self.min_confidence})")
            
            result = VerificationResult(
                is_verified=is_verified,
                confidence_score=confidence_score,
                cited_sections=valid_citations,
                citation_match_count=len(valid_citations),
                similarity_scores=similarity_scores,
                fallback_message=fallback_message,
                details={
                    'domain': domain,
                    'total_citations_extracted': len(citations),
                    'avg_similarity': avg_similarity,
                    'citation_score': citation_score,
                    'overlap_score': overlap_score,
                    'retrieved_chunk_count': len(retrieved_chunks)
                }
            )
            
            logger.info(f"✅ Answer verification complete: "
                       f"verified={is_verified}, confidence={confidence_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during answer verification: {e}")
            # Return low-confidence result on error
            return VerificationResult(
                is_verified=False,
                confidence_score=0.0,
                cited_sections=[],
                citation_match_count=0,
                similarity_scores=[],
                fallback_message=self.get_fallback_message(),
                details={'error': str(e)}
            )
    
    def get_confidence_category(self, confidence_score: float) -> str:
        """
        Get human-readable confidence category.
        
        Args:
            confidence_score: Numeric confidence score (0-1)
            
        Returns:
            Category string: 'Very High', 'High', 'Medium', 'Low', 'Very Low'
        """
        if confidence_score >= 0.85:
            return "Very High"
        elif confidence_score >= 0.70:
            return "High"
        elif confidence_score >= 0.55:
            return "Medium"
        elif confidence_score >= 0.40:
            return "Low"
        else:
            return "Very Low"
