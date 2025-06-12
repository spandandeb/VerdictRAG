from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
from transformers import pipeline, AutoTokenizer
import torch
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import sent_tokenize
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrueRAGVERDICT:
    def __init__(self):
        self.summarizer = None
        self.qa_model = None
        self.sentence_model = None
        self.tokenizer = None
        self.vector_store = None
        self.chunks = []
        self.chunk_embeddings = None
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.initialize_models()
   
    def initialize_models(self):
        """Initialize RAG models and components"""
        try:
            print("Initializing True RAG VERDICTRAg models...")
           
            # Sentence transformer for embeddings (RAG component)
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence transformer loaded")
           
            # QA model for answer generation
            self.qa_model = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ QA model loaded")
           
            # Summarizer for fallback
            model_name = "facebook/bart-large-cnn"
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                framework="pt"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("✅ Summarizer loaded")
           
            print("✅ All RAG models initialized successfully!")
            return True
           
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            return False
   
    def extract_legal_text(self, file):
        """Extract and clean text from legal PDFs"""
        try:
            with pdfplumber.open(file) as pdf:
                text = ''
                max_pages = min(100, len(pdf.pages))  # Increased for better coverage
               
                for page_num in range(max_pages):
                    page = pdf.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n{page_text}"
               
                # Clean the text
                text = self.clean_legal_text(text)
                logger.info(f"Extracted {len(text)} characters from {max_pages} pages")
                return text
               
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            return None
   
    def clean_legal_text(self, text):
        """Clean legal document text"""
        # Remove excessive whitespace and normalize
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'[^\w\s.,;:!?()\-\'""/]', ' ', text)
        return text.strip()
   
    def chunk_document(self, text: str) -> List[Dict]:
        """
        RAG Step 1: Chunk the legal document into overlapping segments
        """
        try:
            # Sentence-based chunking for better semantic coherence
            sentences = sent_tokenize(text)
            chunks = []
            chunk_id = 0
           
            current_chunk = ""
            current_sentences = []
           
            for i, sentence in enumerate(sentences):
                # Clean sentence
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Add sentence to current chunk
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence
               
                # Check if chunk exceeds size limit
                if len(test_chunk.split()) > self.chunk_size:
                    if current_chunk:  # Save current chunk
                        chunks.append({
                            'id': chunk_id,
                            'text': current_chunk.strip(),
                            'sentences': current_sentences.copy(),
                            'start_sentence': i - len(current_sentences),
                            'end_sentence': i - 1
                        })
                        chunk_id += 1
                   
                    # Start new chunk with overlap
                    overlap_sentences = current_sentences[-self.chunk_overlap:] if len(current_sentences) > self.chunk_overlap else current_sentences
                    current_chunk = " ".join(overlap_sentences) + " " + sentence
                    current_sentences = overlap_sentences + [sentence]
                else:
                    current_chunk = test_chunk
                    current_sentences.append(sentence)
           
            # Add final chunk
            if current_chunk.strip():
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'sentences': current_sentences,
                    'start_sentence': len(sentences) - len(current_sentences),
                    'end_sentence': len(sentences) - 1
                })
           
            # Filter out empty chunks
            chunks = [chunk for chunk in chunks if chunk['text'].strip()]
           
            logger.info(f"Document chunked into {len(chunks)} segments")
            return chunks
           
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            # Fallback: simple word-based chunking
            words = text.split()
            chunks = []
            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = ' '.join(chunk_words)
                if chunk_text.strip():  # Only add non-empty chunks
                    chunks.append({
                        'id': i // (self.chunk_size - self.chunk_overlap),
                        'text': chunk_text,
                        'word_start': i,
                        'word_end': i + len(chunk_words)
                    })
            return chunks
   
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """
        RAG Step 2: Create sentence embeddings for all chunks
        """
        try:
            # Filter out empty chunks before creating embeddings
            valid_chunks = [chunk for chunk in chunks if chunk['text'].strip()]
            if not valid_chunks:
                logger.error("No valid chunks found for embedding creation")
                return None
                
            texts = [chunk['text'] for chunk in valid_chunks]
            embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
            logger.info(f"Created embeddings for {len(valid_chunks)} chunks, shape: {embeddings.shape}")
            return embeddings
           
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return None
   
    def build_vector_store(self, embeddings: np.ndarray):
        """
        RAG Step 3: Store embeddings in FAISS vector store
        """
        try:
            # Initialize FAISS index
            dimension = embeddings.shape[1]
            self.vector_store = faiss.IndexFlatIP(dimension)  # Inner product for similarity
           
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
           
            # Add embeddings to index
            self.vector_store.add(embeddings.astype('float32'))
           
            logger.info(f"Built FAISS vector store with {self.vector_store.ntotal} vectors")
            return True
           
        except Exception as e:
            logger.error(f"Error building vector store: {e}")
            return False
   
    def retrieve_relevant_chunks(self, question: str, k: int = 5) -> List[Dict]:
        """
        RAG Step 4: Embed user question and retrieve similar chunks
        """
        try:
            # Embed the question
            question_embedding = self.sentence_model.encode([question])
            faiss.normalize_L2(question_embedding)
           
            # Search for similar chunks
            scores, indices = self.vector_store.search(question_embedding.astype('float32'), k)
           
            # Retrieve relevant chunks with scores
            retrieved_chunks = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and idx < len(self.chunks):  # Valid index check
                    chunk = self.chunks[idx].copy()
                    chunk['similarity_score'] = float(score)
                    chunk['rank'] = i + 1
                    retrieved_chunks.append(chunk)
           
            logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks for question: '{question[:50]}...'")
            return retrieved_chunks
           
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
   
    def generate_answer_from_chunks(self, question: str, retrieved_chunks: List[Dict]) -> Dict:
        """
        RAG Step 5: Pass retrieved chunks to QA model for answer generation
        """
        try:
            if not retrieved_chunks:
                return {
                    "answer": "I couldn't find relevant information in the document to answer your question.",
                    "confidence": 0.0,
                    "source_chunks": 0,
                    "retrieved_chunks": 0,
                    "context_length": 0,
                    "chunk_sources": []
                }
           
            # Combine top chunks as context (limit to avoid token overflow)
            context_parts = []
            total_length = 0
            max_context_length = 3000  # Leave room for question and model processing
           
            for chunk in retrieved_chunks:
                chunk_text = chunk['text'].strip()
                if not chunk_text:  # Skip empty chunks
                    continue
                    
                if total_length + len(chunk_text) < max_context_length:
                    context_parts.append(f"[Chunk {chunk['rank']}]: {chunk_text}")
                    total_length += len(chunk_text)
                else:
                    break
           
            # Check if we have valid context
            if not context_parts:
                return {
                    "answer": "The retrieved document sections appear to be empty or contain no readable text.",
                    "confidence": 0.0,
                    "source_chunks": 0,
                    "retrieved_chunks": len(retrieved_chunks),
                    "context_length": 0,
                    "chunk_sources": []
                }
            
            combined_context = "\n\n".join(context_parts)
            
            # Final check for empty context
            if not combined_context.strip():
                return {
                    "answer": "No readable content found in the relevant document sections.",
                    "confidence": 0.0,
                    "source_chunks": 0,
                    "retrieved_chunks": len(retrieved_chunks),
                    "context_length": 0,
                    "chunk_sources": []
                }
           
            # Generate answer using QA model
            result = self.qa_model(question=question, context=combined_context)
           
            # Enhance result with chunk information
            enhanced_result = {
                "answer": result['answer'],
                "confidence": result['score'],
                "source_chunks": len(context_parts),
                "retrieved_chunks": len(retrieved_chunks),
                "context_length": len(combined_context),
                "chunk_sources": [
                    {
                        "chunk_id": chunk['id'],
                        "rank": chunk['rank'],
                        "similarity_score": chunk['similarity_score'],
                        "preview": chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
                    }
                    for chunk in retrieved_chunks[:3]  # Top 3 sources
                ]
            }
           
            logger.info(f"Generated answer with confidence {result['score']:.3f} using {len(context_parts)} chunks")
            return enhanced_result
           
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "confidence": 0.0,
                "source_chunks": 0,
                "retrieved_chunks": len(retrieved_chunks) if retrieved_chunks else 0,
                "context_length": 0,
                "chunk_sources": []
            }
   
    def process_document_with_rag(self, text: str) -> bool:
        """
        Complete RAG pipeline: Process document and build searchable index
        """
        try:
            logger.info("Starting RAG document processing pipeline...")
           
            # Validate input text
            if not text or not text.strip():
                logger.error("Input text is empty")
                return False
           
            # Step 1: Chunk the document
            self.chunks = self.chunk_document(text)
            if not self.chunks:
                logger.error("Failed to chunk document - no valid chunks created")
                return False
           
            # Step 2: Create embeddings
            self.chunk_embeddings = self.create_embeddings(self.chunks)
            if self.chunk_embeddings is None:
                logger.error("Failed to create embeddings")
                return False
           
            # Step 3: Build vector store
            if not self.build_vector_store(self.chunk_embeddings):
                logger.error("Failed to build vector store")
                return False
           
            logger.info("RAG pipeline completed successfully!")
            return True
           
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return False
   
    def rag_query(self, question: str, k: int = 5) -> Dict:
        """
        Main RAG query function: Retrieve and generate answer
        """
        try:
            if not self.vector_store or not self.chunks:
                return {"error": "Document not processed. Please upload and process a document first."}
           
            # Validate question
            if not question or not question.strip():
                return {"error": "Question cannot be empty."}
           
            # Steps 4 & 5: Retrieve and generate
            retrieved_chunks = self.retrieve_relevant_chunks(question, k)
            answer_result = self.generate_answer_from_chunks(question, retrieved_chunks)
           
            return {
                "question": question,
                "answer": answer_result["answer"],
                "confidence": answer_result["confidence"],
                "rag_metadata": {
                    "total_chunks": len(self.chunks),
                    "retrieved_chunks": answer_result.get("retrieved_chunks", 0),
                    "source_chunks_used": answer_result.get("source_chunks", 0),
                    "context_length": answer_result.get("context_length", 0),
                    "chunk_sources": answer_result.get("chunk_sources", [])
                }
            }
           
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return {"error": f"RAG query failed: {str(e)}"}
   
    def simplify_legal_language(self, text):
        """Replace complex legal terms with simpler alternatives"""
        replacements = {
            'plaintiff': 'the person who filed the lawsuit',
            'defendant': 'the person being sued',
            'appellant': 'the party appealing the decision',
            'respondent': 'the party responding to the appeal',
            'petitioner': 'the person who filed the petition',
            'prima facie': 'at first glance',
            'inter alia': 'among other things',
            'ex parte': 'one-sided',
            'bona fide': 'genuine',
            'ultra vires': 'beyond legal authority',
            'res judicata': 'matter already decided',
            'mandamus': 'court order to perform duty',
            'certiorari': 'review by higher court',
            'habeas corpus': 'right to be released from unlawful detention'
        }
       
        simplified_text = text
        for legal_term, simple_term in replacements.items():
            simplified_text = re.sub(
                rf'\b{re.escape(legal_term)}\b',
                simple_term,
                simplified_text,
                flags=re.IGNORECASE
            )
       
        return simplified_text
   
    def extract_case_information(self, text):
        """Extract basic case information"""
        case_info = {}
        sample_text = text[:5000]
        lines = sample_text.split('\n')[:20]
       
        # Case title
        for line in lines:
            if any(vs in line.lower() for vs in [' vs ', ' v. ', ' v ', ' versus ']):
                case_info['title'] = line.strip()
                break
       
        # Citation patterns
        citation_patterns = [
            r'AIR\s*\d{4}.*?\d+',
            r'\d{4}\s*\(\d+\)\s*[A-Z]+\s*\d+',
        ]
       
        for pattern in citation_patterns:
            match = re.search(pattern, sample_text)
            if match:
                case_info['citation'] = match.group(0)
                break
       
        return case_info

# Initialize the RAG system
rag_verdict = TrueRAGVERDICT()

@app.route("/upload", methods=["POST"])
def upload_and_process_document():
    """Upload document and build RAG index"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        logger.info(f"Processing file: {file.filename}")

        # Extract text
        legal_text = rag_verdict.extract_legal_text(file)
        if not legal_text:
            return jsonify({"error": "Failed to extract text from legal PDF"}), 400

        if len(legal_text.strip()) < 100:
            return jsonify({"error": "Document appears to be empty or corrupted"}), 400

        # Process with RAG pipeline
        if not rag_verdict.process_document_with_rag(legal_text):
            return jsonify({"error": "Failed to process document with RAG pipeline"}), 500

        # Extract case information
        case_info = rag_verdict.extract_case_information(legal_text)

        return jsonify({
            "message": "Document processed successfully with RAG pipeline",
            "case_info": case_info,
            "document_stats": {
                "total_characters": len(legal_text),
                "total_words": len(legal_text.split()),
                "total_chunks": len(rag_verdict.chunks),
                "chunk_size": rag_verdict.chunk_size,
                "chunk_overlap": rag_verdict.chunk_overlap,
                "embedding_dimension": rag_verdict.chunk_embeddings.shape[1] if rag_verdict.chunk_embeddings is not None else 0
            },
            "rag_status": "Ready for queries"
        })

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return jsonify({"error": f"Document processing error: {str(e)}"}), 500

@app.route("/query", methods=["POST"])
def rag_query():
    """Query the processed document using RAG"""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400

        question = data['question']
        k = data.get('k', 5)  # Number of chunks to retrieve

        # Perform RAG query
        result = rag_verdict.rag_query(question, k)
        
        if "error" in result:
            return jsonify(result), 400

        # Simplify legal language in answer
        result['answer'] = rag_verdict.simplify_legal_language(result['answer'])

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in query: {e}")
        return jsonify({"error": f"Query error: {str(e)}"}), 500

@app.route("/analyze", methods=["POST"])
def comprehensive_analysis():
    """Comprehensive analysis using multiple RAG queries"""
    try:
        if not rag_verdict.vector_store or not rag_verdict.chunks:
            return jsonify({"error": "No document processed. Please upload a document first."}), 400

        # Predefined analysis questions
        analysis_questions = [
            "What is this case about and what are the main facts?",
            "Who are the parties involved in this case?",
            "What are the main legal issues or questions in this case?",
            "What did the court decide and what was the reasoning?",
            "What is the significance or impact of this case?"
        ]

        analysis_results = {}
        
        for question in analysis_questions:
            result = rag_verdict.rag_query(question, k=3)
            if "error" not in result:
                key = question.split('?')[0].lower().replace(' ', '_')
                analysis_results[key] = {
                    "question": question,
                    "answer": result['answer'],
                    "confidence": result['confidence'],
                    "sources": result['rag_metadata']['chunk_sources']
                }

        return jsonify({
            "comprehensive_analysis": analysis_results,
            "document_stats": {
                "total_chunks": len(rag_verdict.chunks),
                "embedding_dimension": rag_verdict.chunk_embeddings.shape[1] if rag_verdict.chunk_embeddings is not None else 0
            },
            "system": "True RAG VERDICTRAg - Retrieval Augmented Generation"
        })

    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        return jsonify({"error": f"Analysis error: {str(e)}"}), 500

@app.route("/chunks", methods=["GET"])
def get_chunks():
    """Get information about document chunks"""
    try:
        if not rag_verdict.chunks:
            return jsonify({"error": "No document processed"}), 400

        chunk_info = []
        for i, chunk in enumerate(rag_verdict.chunks[:10]):  # First 10 chunks
            chunk_info.append({
                "id": chunk['id'],
                "preview": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                "length": len(chunk['text']),
                "word_count": len(chunk['text'].split())
            })

        return jsonify({
            "total_chunks": len(rag_verdict.chunks),
            "sample_chunks": chunk_info,
            "chunk_size": rag_verdict.chunk_size,
            "chunk_overlap": rag_verdict.chunk_overlap
        })

    except Exception as e:
        logger.error(f"Error getting chunks: {e}")
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "True RAG VERDICTRAg Online",
        "version": "6.0 - Retrieval Augmented Generation",
        "rag_components": {
            "chunking": "✅ Semantic sentence-based chunking",
            "embeddings": "✅ SentenceTransformer embeddings",
            "vector_store": "✅ FAISS similarity search",
            "retrieval": "✅ Top-k chunk retrieval",
            "generation": "✅ Context-aware QA"
        },
        "models": {
            "sentence_model": rag_verdict.sentence_model is not None,
            "qa_model": rag_verdict.qa_model is not None,
            "summarizer": rag_verdict.summarizer is not None,
            "vector_store": rag_verdict.vector_store is not None
        },
        "document_status": {
            "chunks_loaded": len(rag_verdict.chunks) if rag_verdict.chunks else 0,
            "embeddings_ready": rag_verdict.chunk_embeddings is not None,
            "vector_store_ready": rag_verdict.vector_store is not None
        }
    })

if __name__ == "__main__":
    print("🚀 Starting True RAG VERDICTRAg - Retrieval Augmented Generation")
    print("📊 RAG Pipeline: Chunking → Embeddings → Vector Store → Retrieval → Generation")
    print("🔍 Features: FAISS Vector Search, Semantic Chunking, Context-Aware QA")
    app.run(debug=True, port=5000)