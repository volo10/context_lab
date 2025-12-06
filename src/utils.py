"""
Utility Functions Module
========================
Provides text generation, document manipulation, and embedding utilities.

Domain-Specific Text Generation:
- Medical Hebrew (רפואי)
- Technology Hebrew (טכנולוגיה)
- Legal Hebrew (משפטי)
- General English (placeholder)
"""

import logging
import random
from typing import List

import numpy as np

from .llm import REAL_LLM_AVAILABLE, get_embeddings

logger = logging.getLogger(__name__)

# Domain-specific filler phrase banks
FILLER_PHRASES = {
    "medical_hebrew": [
        "מחקרים קליניים הראו כי טיפול תרופתי משפר את המצב הרפואי של המטופלים.",
        "בדיקות מעבדה מצביעות על שיפור משמעותי בתפקוד האיברים הפנימיים.",
        "תופעות לוואי נדירות עשויות להתרחש במהלך הטיפול התרופתי.",
        "יש להתייעץ עם רופא לפני נטילת התרופה במקרים של מחלות רקע.",
        "המינון המומלץ הוא בהתאם להנחיות הרופא המטפל.",
        "במקרה של תגובה אלרגית יש לפנות מיד לחדר מיון.",
        "הטיפול מיועד למבוגרים וילדים מעל גיל שתים עשרה.",
        "אחסון התרופה צריך להיות בטמפרטורת החדר הרגילה.",
        "תרופה זו משמשת לטיפול בכאבים ודלקות בדרגות שונות.",
        "יעילות הטיפול נבדקה במחקרים רבים ברחבי העולם.",
    ],
    "tech_hebrew": [
        "הטכנולוגיה החדשה מאפשרת עיבוד נתונים במהירות חסרת תקדים.",
        "מערכות בינוי מלאכותית משתפרות בקצב מהיר ביותר בשנים האחרונות.",
        "אבטחת מידע היא אחד האתגרים המרכזיים בעידן הדיגיטלי.",
        "פיתוח תוכנה מודרנית דורש כלים וטכנולוגיות מתקדמות.",
        "מחשוב ענן מספק פתרונות גמישים לעסקים בכל הגדלים.",
        "אלגוריתמים מתקדמים מאפשרים ניתוח מדויק של מידע רב.",
        "רשתות תקשורת מהירות הן הבסיס לחברה המקוונת.",
        "ניהול מסדי נתונים גדולים דורש תשתיות חזקות ויעילות.",
        "פתרונות ענן מספקים גמישות ויכולת התרחבות למערכות מידע.",
        "אבטחת סייבר היא תחום קריטי בעידן המודרני.",
    ],
    "legal_hebrew": [
        "בית המשפט קבע כי הראיות המוצגות אינן מספקות להרשעה.",
        "התובע הגיש ערעור על פסק הדין שניתן בערכאה הראשונה.",
        "החוק קובע כי יש לפעול בהתאם לכללי הצדק והיושר.",
        "הנתבע טען כי לא היה מודע להשלכות המשפטיות של מעשיו.",
        "בית הדין הכריע בסכסוך בין הצדדים באופן סופי ומחייב.",
        "הסדר הפשרה אושר על ידי בית המשפט לאחר דיונים ממושכים.",
        "הפסיקה הנוכחית מתבססת על תקדימים משפטיים קודמים.",
        "זכויות הנאשם נשמרות במהלך כל שלבי ההליך המשפטי.",
        "החלטת בית המשפט תקפה מיום מתן פסק הדין.",
        "הצדדים התבקשו להגיש סיכומים בכתב בתוך שלושים יום.",
    ],
    "general": [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
        "Duis aute irure dolor in reprehenderit in voluptate velit esse.",
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa.",
    ],
}


def generate_filler_text(num_words: int = 200, domain: str = "general") -> str:
    """
    Generate domain-specific filler text.

    Supports multiple domains for realistic document simulation:
    - medical_hebrew: Clinical/pharmaceutical terminology
    - tech_hebrew: Technology and computing terms
    - legal_hebrew: Legal and judicial terminology
    - general: Lorem ipsum placeholder

    Args:
        num_words: Target word count
        domain: Text domain (medical_hebrew, tech_hebrew, legal_hebrew, general)

    Returns:
        Generated filler text string
    """
    phrases = FILLER_PHRASES.get(domain, FILLER_PHRASES["general"])

    words = []
    while len(words) < num_words:
        phrase = random.choice(phrases)
        words.extend(phrase.split())

    result = " ".join(words[:num_words])
    logger.debug(f"Generated {num_words} words of {domain} filler text")
    return result


def embed_critical_fact(doc_text: str, fact: str, position: str) -> str:
    """
    Embed a critical fact at a specific position within document text.

    Position Mapping (normalized to document length):
    - start: 5% into document (position = 0.05)
    - middle: 50% into document (position = 0.50)
    - end: 95% into document (position = 0.95)

    This positioning is crucial for "Lost in the Middle" experiments:
    - Research shows LLMs have U-shaped recall curves
    - Facts at start/end have ~90% retrieval rate
    - Facts in middle have ~50% retrieval rate

    Args:
        doc_text: Base document text
        fact: Critical fact to embed
        position: Placement ('start', 'middle', 'end')

    Returns:
        Document with embedded fact

    Raises:
        ValueError: If position is not valid
    """
    words = doc_text.split()
    fact_words = fact.split()

    position_map = {
        'start': len(words) // 20,        # 5%
        'middle': len(words) // 2,         # 50%
        'end': (len(words) * 19) // 20,    # 95%
    }

    if position not in position_map:
        raise ValueError(f"Invalid position: {position}. Must be 'start', 'middle', or 'end'")

    insert_idx = position_map[position]

    # Insert fact words at calculated position
    for i, word in enumerate(fact_words):
        words.insert(insert_idx + i, word)

    logger.debug(f"Embedded fact at {position} (index {insert_idx}/{len(words)})")
    return " ".join(words)


def split_documents(documents: List[str], chunk_size: int = 500) -> List[str]:
    """
    Split documents into chunks for embedding.

    Chunking Strategy:
    - Fixed-size character chunks with word boundaries
    - No overlap (can be extended for sliding window)
    - Preserves word integrity

    For RAG systems, optimal chunk size depends on:
    - Embedding model context window
    - Query granularity
    - Retrieval precision requirements

    Typical ranges: 200-1000 characters

    Args:
        documents: List of document strings
        chunk_size: Target chunk size in characters

    Returns:
        List of text chunks
    """
    chunks = []

    for doc in documents:
        words = doc.split()
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space

            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

        # Don't lose remaining words
        if current_chunk:
            chunks.append(" ".join(current_chunk))

    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def load_documents(
    num_docs: int,
    words_per_doc: int = 200,
    diverse_domains: bool = False
) -> List[str]:
    """
    Generate or load a set of documents.

    Generation Modes:
    1. Single domain (diverse_domains=False): All general/English
    2. Multi-domain (diverse_domains=True): Rotate through Hebrew domains

    Domain rotation ensures balanced corpus:
    - Medical: ~33%
    - Technology: ~33%
    - Legal: ~33%

    Args:
        num_docs: Number of documents to generate
        words_per_doc: Words per document
        diverse_domains: Enable Hebrew multi-domain generation

    Returns:
        List of document strings
    """
    if diverse_domains:
        domains = ["medical_hebrew", "tech_hebrew", "legal_hebrew"]
        documents = []

        for i in range(num_docs):
            domain = domains[i % len(domains)]
            documents.append(generate_filler_text(words_per_doc, domain=domain))

        logger.info(f"Generated {num_docs} diverse Hebrew documents")
        return documents

    logger.info(f"Generated {num_docs} general documents")
    return [generate_filler_text(words_per_doc) for _ in range(num_docs)]


def concatenate_documents(documents: List[str]) -> str:
    """
    Concatenate documents into a single context string.

    Separator: "---" with newlines for clear document boundaries
    This helps LLMs understand document structure.

    Args:
        documents: List of document strings

    Returns:
        Single concatenated context string
    """
    result = "\n\n---\n\n".join(documents)
    logger.debug(f"Concatenated {len(documents)} documents ({len(result)} chars)")
    return result


def nomic_embed_text(chunks: List[str], use_real: bool = None) -> List[np.ndarray]:
    """
    Generate embeddings for text chunks.

    Embedding Models:
    1. Hebrew text: paraphrase-multilingual-MiniLM-L12-v2 (384d)
       - Supports 50+ languages including Hebrew
       - Semantic similarity across languages

    2. English text: nomic-embed-text via Ollama or all-MiniLM-L6-v2 (384d)
       - Optimized for English semantic search

    3. Fallback: Random vectors (simulation mode)

    Args:
        chunks: Text chunks to embed
        use_real: Force real/simulation mode

    Returns:
        List of embedding vectors (384-dimensional)
    """
    if use_real is None:
        use_real = REAL_LLM_AVAILABLE

    if use_real and REAL_LLM_AVAILABLE:
        # Detect Hebrew in first few chunks
        has_hebrew = any(
            any('\u0590' <= c <= '\u05FF' for c in chunk)
            for chunk in chunks[:5]
        )

        if has_hebrew:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Using multilingual embeddings for Hebrew text")
                model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                embeddings = model.encode(chunks, show_progress_bar=False)
                return embeddings.tolist()
            except Exception as e:
                logger.warning(f"Multilingual embeddings failed: {e}")

        # Try Ollama embeddings for English
        embeddings_model = get_embeddings()
        if embeddings_model and not has_hebrew:
            try:
                return embeddings_model.embed_documents(chunks)
            except Exception as e:
                logger.warning(f"Ollama embeddings failed: {e}")

        # Fallback to standard sentence transformers
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(chunks, show_progress_bar=False).tolist()
        except Exception as e:
            logger.warning(f"Sentence transformers failed: {e}")

    # Simulation mode: random 384-dim vectors
    logger.debug(f"Generating {len(chunks)} simulated embeddings")
    return [np.random.randn(384).tolist() for _ in chunks]
