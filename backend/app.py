from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
from transformers import pipeline, AutoTokenizer
import torch
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedVERDICTRAg:
    def __init__(self):
        self.summarizer = None
        self.qa_model = None
        self.sentence_model = None
        self.tokenizer = None
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize lightweight models for faster processing"""
        try:
            print("Initializing Optimized VERDICTRAg models...")
            
            # Use faster, smaller models
            model_name = "facebook/bart-large-cnn"
            self.summarizer = pipeline(
                "summarization", 
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                framework="pt"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Faster QA model
            self.qa_model = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Lightweight sentence transformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            print("✅ All models initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            return False
    
    def extract_legal_text(self, file):
        """Extract and clean text from legal PDFs with size limits"""
        try:
            with pdfplumber.open(file) as pdf:
                text = ''
                # Limit to first 50 pages for faster processing
                max_pages = min(50, len(pdf.pages))
                
                for page_num in range(max_pages):
                    page = pdf.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n{page_text}"
                    
                    # Break if we have enough text (limit to ~200K chars)
                    if len(text) > 200000:
                        break
                
                # Clean the text
                text = self.clean_legal_text(text)
                logger.info(f"Extracted {len(text)} characters from {max_pages} pages")
                return text
                
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            return None
    
    def clean_legal_text(self, text):
        """Clean legal document text efficiently"""
        # Basic cleaning only
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()
    
    def extract_case_information(self, text):
        """Extract basic case information from first 5000 chars only"""
        case_info = {}
        
        # Only check first 5000 characters for speed
        sample_text = text[:5000]
        lines = sample_text.split('\n')[:15]  # First 15 lines only
        
        # Case title
        for line in lines:
            if any(vs in line.lower() for vs in [' vs ', ' v. ', ' v ', ' versus ']):
                case_info['title'] = line.strip()
                break
        
        # Simple citation patterns
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
    
    def simplify_legal_language(self, text):
        """Replace complex legal terms with simpler alternatives"""
        replacements = {
            'plaintiff': 'the person who filed the lawsuit',
            'defendant': 'the person being sued',
            'appellant': 'the party appealing the decision',
            'respondent': 'the party responding to the appeal',
            'petitioner': 'the person who filed the petition',
            'transferee': 'the person receiving the transfer',
            'transferor': 'the person making the transfer',
            'equity shares': 'company shares',
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
    
    def extract_parties_simple(self, text):
        """Extract and explain the parties in simple terms"""
        try:
            # Look for party information
            parties_queries = [
                "Who is the plaintiff?",
                "Who is the defendant?",
                "Who are the main parties?"
            ]
            
            parties_info = []
            for query in parties_queries:
                try:
                    result = self.qa_model(question=query, context=text[:4000])
                    if result and result.get('score', 0) > 0.1:
                        answer = self.simplify_legal_language(result['answer'])
                        if len(answer.strip()) > 10:
                            parties_info.append(answer)
                except:
                    continue
            
            if parties_info:
                unique_parties = list(set(parties_info))
                return "The main parties in this case are: " + "; ".join(unique_parties[:3])
            
            return None
        except:
            return None
    
    def extract_case_story(self, text):
        """Extract the narrative of what happened"""
        try:
            story_queries = [
                "What are the facts of this case?",
                "What happened that led to this dispute?",
                "What events caused this lawsuit?"
            ]
            
            for query in story_queries:
                try:
                    result = self.qa_model(question=query, context=text[:5000])
                    if result and result.get('score', 0) > 0.15:
                        story = self.simplify_legal_language(result['answer'])
                        # Clean up the story
                        if len(story.strip()) > 50:
                            return story
                except:
                    continue
            
            return None
        except:
            return None
    
    def extract_legal_issue_simple(self, text):
        """Extract the legal issue in simple terms"""
        try:
            issue_queries = [
                "What is the main legal issue?",
                "What law is being disputed?",
                "What legal question needs to be answered?"
            ]
            
            for query in issue_queries:
                try:
                    result = self.qa_model(question=query, context=text[:4000])
                    if result and result.get('score', 0) > 0.15:
                        issue = self.simplify_legal_language(result['answer'])
                        if len(issue.strip()) > 20:
                            return f"The main legal question was: {issue}"
                except:
                    continue
            
            return None
        except:
            return None
    
    def extract_court_decision_simple(self, text):
        """Extract the court's decision in plain English"""
        try:
            decision_queries = [
                "What did the court decide?",
                "Who won the case?",
                "What was the outcome?",
                "What was the judgment?"
            ]
            
            # Look at the end of the document for decision
            decision_text = text[-8000:] if len(text) > 8000 else text
            
            for query in decision_queries:
                try:
                    result = self.qa_model(question=query, context=decision_text)
                    if result and result.get('score', 0) > 0.15:
                        decision = self.simplify_legal_language(result['answer'])
                        if len(decision.strip()) > 20:
                            return decision
                except:
                    continue
            
            return None
        except:
            return None
    
    def extract_case_impact(self, text):
        """Explain why this case matters"""
        try:
            impact_queries = [
                "What is the significance of this case?",
                "What does this decision mean?",
                "What precedent does this set?"
            ]
            
            for query in impact_queries:
                try:
                    result = self.qa_model(question=query, context=text[-5000:])
                    if result and result.get('score', 0) > 0.1:
                        impact = self.simplify_legal_language(result['answer'])
                        if len(impact.strip()) > 20:
                            return impact
                except:
                    continue
            
            # Fallback explanation
            return "This case helps establish legal precedent and provides guidance for similar future disputes."
        except:
            return None
    
    def create_medium_summary(self, text, max_length=300):
        """Create a medium-length summary with robust error handling"""
        try:
            # Clean and limit input text
            input_text = text.strip()[:6000]  # Reduced size for safety
            
            # Ensure minimum length
            if len(input_text) < 100:
                return "Document too short for meaningful summary."
            
            # Calculate token count and adjust
            tokens = self.tokenizer.encode(input_text, truncation=True, max_length=1024)
            if len(tokens) < 50:
                return "Insufficient content for summary generation."
            
            # Safe summarization with multiple fallbacks
            try:
                summary = self.summarizer(
                    input_text,
                    max_length=min(max_length, 512),  # Cap at model limit
                    min_length=min(50, max_length//3),  # Ensure reasonable min
                    do_sample=False,
                    truncation=True,
                    clean_up_tokenization_spaces=True
                )[0]['summary_text']
                
                return self.simplify_legal_language(summary)
                
            except Exception as model_error:
                logger.warning(f"Model summarization failed: {model_error}")
                # Fallback to extractive summary
                return self.create_extractive_summary(input_text, max_length)
            
        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            return self.create_extractive_summary(text[:3000], max_length)
    
    def create_extractive_summary(self, text, max_words=300):
        """Fallback extractive summary when model fails"""
        try:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Select important sentences (first, last, and middle parts)
            important_sentences = []
            
            if len(sentences) > 0:
                important_sentences.append(sentences[0])  # First sentence
            
            if len(sentences) > 5:
                mid_point = len(sentences) // 2
                important_sentences.extend(sentences[mid_point:mid_point+2])  # Middle sentences
            
            if len(sentences) > 2:
                important_sentences.append(sentences[-1])  # Last sentence
            
            # Join and limit by word count
            summary = ' '.join(important_sentences)
            words = summary.split()
            
            if len(words) > max_words:
                summary = ' '.join(words[:max_words]) + "..."
            
            simplified_summary = self.simplify_legal_language(summary) if summary else "Unable to create summary from document content."
            
            return simplified_summary
            
        except Exception as e:
            logger.error(f"Extractive summary failed: {e}")
            return "Summary generation failed. Document may be corrupted or empty."
    
    def create_comprehensive_summary(self, text):
        """Create a comprehensive but accessible summary"""
        try:
            # Use the middle portion of the document for comprehensive view
            middle_start = len(text) // 4
            middle_end = 3 * len(text) // 4
            middle_text = text[middle_start:middle_end][:6000]
            
            summary = self.create_medium_summary(middle_text, max_length=400)
            return summary
        except:
            return self.create_extractive_summary(text[:5000], 300)
    
    def create_fallback_analysis(self, text):
        """Fallback analysis when main extraction fails"""
        try:
            case_info = self.extract_case_information(text)
            summary = self.create_comprehensive_summary(text)
            
            fallback = "## LEGAL CASE ANALYSIS\n\n"
            
            if case_info and 'title' in case_info:
                title = case_info['title'].replace(' vs ', ' versus ')
                fallback += f"**Case:** {title}\n\n"
            
            fallback += f"**Summary:** {summary}\n\n"
            fallback += "**Note:** This is a simplified analysis designed for non-lawyers. For detailed legal advice, please consult with a qualified attorney.\n"
            
            return fallback
        except:
            return "## ANALYSIS UNAVAILABLE\n\nUnable to process this legal document. Please ensure the file is readable and try again."
    
    def create_user_friendly_analysis(self, text):
        """Create analysis specifically for users with no legal background"""
        try:
            analysis_sections = []
            
            # 1. Simple Case Overview
            case_info = self.extract_case_information(text)
            
            if case_info:
                header = "## CASE OVERVIEW\n\n"
                if 'title' in case_info:
                    # Make title more readable
                    title = case_info['title'].replace(' vs ', ' versus ')
                    header += f"**Case:** {title}\n"
                if 'citation' in case_info:
                    header += f"**Citation:** {case_info['citation']}\n"
                header += "\n"
                analysis_sections.append(header)
            
            # 2. What This Case Is About (Plain English)
            try:
                # Extract main dispute in simple terms
                context_queries = [
                    "What is this case about?",
                    "What is the main dispute?",
                    "Why did this go to court?"
                ]
                
                case_context = None
                for query in context_queries:
                    try:
                        result = self.qa_model(question=query, context=text[:3000])
                        if result and result.get('score', 0) > 0.15:
                            case_context = self.simplify_legal_language(result['answer'])
                            break
                    except:
                        continue
                
                if case_context and len(case_context.strip()) > 30:
                    analysis_sections.append(f"## WHAT THIS CASE IS ABOUT\n\n{case_context}\n\n")
            except Exception as e:
                logger.warning(f"Case context extraction failed: {e}")
            
            # 3. The Parties Involved (Who's Who)
            try:
                parties_info = self.extract_parties_simple(text)
                if parties_info:
                    analysis_sections.append(f"## WHO'S INVOLVED\n\n{parties_info}\n\n")
            except Exception as e:
                logger.warning(f"Parties extraction failed: {e}")
            
            # 4. What Happened (The Story)
            try:
                story = self.extract_case_story(text)
                if story:
                    analysis_sections.append(f"## WHAT HAPPENED\n\n{story}\n\n")
            except Exception as e:
                logger.warning(f"Story extraction failed: {e}")
            
            # 5. The Legal Issue (Simplified)
            try:
                legal_issue = self.extract_legal_issue_simple(text)
                if legal_issue:
                    analysis_sections.append(f"## THE LEGAL QUESTION\n\n{legal_issue}\n\n")
            except Exception as e:
                logger.warning(f"Legal issue extraction failed: {e}")
            
            # 6. The Court's Decision
            try:
                decision = self.extract_court_decision_simple(text)
                if decision:
                    analysis_sections.append(f"## WHAT THE COURT DECIDED\n\n{decision}\n\n")
            except Exception as e:
                logger.warning(f"Decision extraction failed: {e}")
            
            # 7. Why This Matters (Impact)
            try:
                impact = self.extract_case_impact(text)
                if impact:
                    analysis_sections.append(f"## WHY THIS CASE MATTERS\n\n{impact}\n\n")
            except Exception as e:
                logger.warning(f"Impact extraction failed: {e}")
            
            final_analysis = ''.join(analysis_sections)
            
            # Ensure it's comprehensive but not overwhelming
            if len(final_analysis.split()) < 200:
                # Add fallback comprehensive summary
                comprehensive_summary = self.create_comprehensive_summary(text)
                final_analysis += f"## COMPREHENSIVE SUMMARY\n\n{comprehensive_summary}\n\n"
            
            # Add disclaimer
            final_analysis += "---\n**Disclaimer:** This analysis is designed for educational purposes and uses simplified language for non-lawyers. For legal advice, please consult a qualified attorney.\n"
            
            return final_analysis if final_analysis else self.create_fallback_analysis(text)
            
        except Exception as e:
            logger.error(f"Error creating user-friendly analysis: {e}")
            return self.create_fallback_analysis(text)
    
    def extract_key_points_fast(self, text):
        """Fast extraction of 3-4 key points with error handling"""
        key_points = {}
        
        # Limit text for faster processing
        sample_text = text[:8000]  # Increased slightly for better context
        
        # Only extract most important points
        important_queries = {
            'main_issue': "What is the main legal issue?",
            'decision': "What did the court decide?",
            'parties': "Who are the parties?"
        }
        
        for point_type, query in important_queries.items():
            try:
                # Safe context limiting
                context = sample_text[:2500]  # Safe context size
                
                if len(context.strip()) < 50:
                    continue
                
                result = self.qa_model(
                    question=query, 
                    context=context
                )
                
                if result and result.get('score', 0) > 0.1:  # Lower threshold but check existence
                    # Limit answer length and clean it
                    answer = self.simplify_legal_language(result['answer'][:150].strip())
                    if len(answer) > 10:  # Ensure meaningful answer
                        key_points[point_type] = answer
                
            except Exception as e:
                logger.warning(f"Error extracting {point_type}: {e}")
                continue
        
        # If no points extracted, add fallback
        if not key_points:
            key_points['summary'] = self.create_extractive_summary(sample_text, 100)
        
        return key_points
    
    def create_concise_analysis(self, text):
        """Create concise, medium-length analysis with robust error handling"""
        try:
            analysis_sections = []
            
            # 1. Case Information (brief)
            case_info = self.extract_case_information(text)
            
            if case_info:
                header = "## CASE OVERVIEW\n\n"
                if 'title' in case_info:
                    title = case_info['title'].replace(' vs ', ' versus ')
                    header += f"**Case:** {title}\n"
                if 'citation' in case_info:
                    header += f"**Citation:** {case_info['citation']}\n"
                header += "\n"
                analysis_sections.append(header)
            
            # 2. Summary (medium length) with error handling
            try:
                summary = self.create_medium_summary(text, max_length=250)
                analysis_sections.append(f"## SUMMARY\n\n{summary}\n")
            except Exception as e:
                logger.error(f"Summary creation failed: {e}")
                # Fallback summary
                fallback_summary = self.create_extractive_summary(text[:3000], 200)
                analysis_sections.append(f"## SUMMARY\n\n{fallback_summary}\n")
            
            # 3. Key Points (limited) with error handling
            try:
                key_points = self.extract_key_points_fast(text)
                
                if key_points:
                    analysis_sections.append("## KEY POINTS\n")
                    
                    point_labels = {
                        'main_issue': '**Main Issue:**',
                        'decision': '**Court Decision:**',
                        'parties': '**Parties:**',
                        'summary': '**Key Information:**'
                    }
                    
                    for point_type, content in key_points.items():
                        if content and len(content.strip()) > 10:
                            label = point_labels.get(point_type, f"**{point_type.title()}:**")
                            analysis_sections.append(f"{label} {content}\n")
            except Exception as e:
                logger.error(f"Key points extraction failed: {e}")
                # Add fallback key point
                analysis_sections.append("## KEY POINTS\n")
                analysis_sections.append(f"**Document Overview:** {self.create_extractive_summary(text[:2000], 100)}\n")
            
            # 4. Brief conclusion (optional)
            if len(text) > 10000:
                try:
                    conclusion = self.create_medium_summary(text[-3000:], max_length=100)
                    if conclusion and len(conclusion) > 20:
                        analysis_sections.append(f"## CONCLUSION\n\n{conclusion}\n")
                except Exception as e:
                    logger.warning(f"Conclusion generation failed: {e}")
            
            final_analysis = '\n'.join(analysis_sections)
            
            # Ensure medium length (500-800 words max)
            words = final_analysis.split()
            if len(words) > 600:  # Reduced from 800 for better UX
                final_analysis = ' '.join(words[:600]) + "\n\n*[Analysis truncated for readability]*"
            
            return final_analysis if final_analysis else "Unable to analyze document content."
            
        except Exception as e:
            logger.error(f"Error creating analysis: {e}")
            # Ultimate fallback
            return f"## DOCUMENT ANALYSIS\n\n{self.create_extractive_summary(text[:5000], 300)}"
    
    def analyze_legal_document(self, question, text):
        """Enhanced analysis with user-friendly output"""
        question_lower = question.lower()
        
        # Check if user wants comprehensive analysis or simple explanation
        if any(word in question_lower for word in ['comprehensive', 'analysis', 'explain', 'understand', 'summary', 'naive', 'non-lawyer', 'layperson']):
            return self.create_user_friendly_analysis(text)
        
        elif any(word in question_lower for word in ['simple', 'basic', 'beginner']):
            return self.create_user_friendly_analysis(text)
        
        elif any(word in question_lower for word in ['facts', 'what happened']):
            story = self.extract_case_story(text)
            if story:
                return f"## WHAT HAPPENED IN THIS CASE\n\n{story}"
            else:
                return f"## KEY FACTS\n\n{self.create_medium_summary(text, max_length=300)}"
        
        elif any(word in question_lower for word in ['decision', 'ruling', 'outcome']):
            decision = self.extract_court_decision_simple(text)
            if decision:
                return f"## COURT DECISION\n\n{decision}"
            else:
                # Focus on decision part
                decision_text = text[-8000:] if len(text) > 8000 else text
                return f"## COURT DECISION\n\n{self.create_medium_summary(decision_text, max_length=300)}"
        
        else:
            # Default to user-friendly analysis for any other request
            return self.create_user_friendly_analysis(text)

# Initialize the optimized system
optimized_verdict = OptimizedVERDICTRAg()

@app.route("/analyze/pdf", methods=["POST"])
def analyze_legal_pdf():
    """Optimized PDF analysis with comprehensive error handling"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        question = request.form.get("question", "Provide a comprehensive analysis for non lawyers")

        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        logger.info(f"Processing file: {file.filename}")

        # Extract legal text with limits
        legal_text = optimized_verdict.extract_legal_text(file)
        if not legal_text:
            return jsonify({"error": "Failed to extract text from legal PDF"}), 400

        if len(legal_text.strip()) < 100:
            return jsonify({"error": "Document appears to be empty or corrupted"}), 400

        logger.info(f"Extracted {len(legal_text)} characters")

        # Create user-friendly analysis with error handling
        try:
            analysis = optimized_verdict.analyze_legal_document(question, legal_text)
        except Exception as analysis_error:
            logger.error(f"Analysis failed: {analysis_error}")
            # Fallback to simple extractive summary
            analysis = f"## DOCUMENT SUMMARY\n\n{optimized_verdict.create_extractive_summary(legal_text[:5000], 400)}"

        # Get basic case information
        try:
            case_info = optimized_verdict.extract_case_information(legal_text)
        except Exception as case_error:
            logger.warning(f"Case info extraction failed: {case_error}")
            case_info = {"note": "Case information could not be extracted"}

        return jsonify({
            "analysis": analysis,
            "case_info": case_info,
            "document_type": "Legal Document",
            "text_length": len(legal_text),
            "word_count": len(legal_text.split()),
            "question": question,
            "analysis_length": len(analysis),
            "analysis_word_count": len(analysis.split()),
            "system": "Enhanced VERDICTRAg v5.0 - User-Friendly Legal Analysis",
            "processing_note": "Analysis optimized for non-lawyers with simplified language and structured explanations"
        })

    except Exception as e:
        logger.error(f"Enhanced VERDICTRAg Error: {e}")
        return jsonify({
            "error": f"Legal analysis error: {str(e)}",
            "fallback_message": "The system encountered an error. Please try with a smaller document or contact support."
        }), 500

@app.route("/analyze/quick", methods=["POST"])
def quick_analysis():
    """Ultra-fast analysis for immediate results"""
    try:
        file = request.files["file"]
        legal_text = optimized_verdict.extract_legal_text(file)
        
        if not legal_text:
            return jsonify({"error": "Failed to extract text"}), 400

        # Ultra-quick summary
        quick_summary = optimized_verdict.create_medium_summary(legal_text[:5000], max_length=200)
        case_info = optimized_verdict.extract_case_information(legal_text)

        return jsonify({
            "quick_summary": quick_summary,
            "case_info": case_info,
            "system": "Quick Legal Analysis - Under 30 seconds"
        })

    except Exception as e:
        logger.error(f"Quick analysis error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "Enhanced VERDICTRAg Online",
        "version": "5.0 - User-Friendly Legal Analysis",
        "performance": "Optimized for 2-3 minute processing",
        "output": "Structured analysis for non-lawyers (600-1000 words)",
        "features": [
            "Simplified legal language",
            "Structured storytelling format", 
            "Plain English explanations",
            "Context-rich analysis"
        ],
        "models": {
            "summarizer": optimized_verdict.summarizer is not None,
            "qa_model": optimized_verdict.qa_model is not None,
            "sentence_model": optimized_verdict.sentence_model is not None
        }
    })

if __name__ == "__main__":
    print("🚀 Starting Enhanced VERDICTRAg - User-Friendly Legal Analysis")
    print("⚡ Features: Plain English, Structured Analysis, Beginner-Friendly")
    print("📖 Optimized for: Non-lawyers, Students, General Public")
    app.run(debug=True, port=5000)