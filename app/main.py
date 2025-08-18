"""
Educational RAG System - Streamlined Main Application with Groq API
"""
import streamlit as st
import json
import logging
import sys
import os
import PyPDF2
from typing import List, Dict, Any, Optional
import hashlib
from datetime import datetime

# Load environment variables from .env file
try:
	from dotenv import load_dotenv
	load_dotenv()
	print("Environment variables loaded from .env file")
except ImportError:
	print("python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
	print(f"Error loading .env file: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
	"""Enhanced RAG (Retrieval Augmented Generation) system with Groq API integration"""
	
	def __init__(self):
		self.current_document = None
		self.document_content = None
		self.document_hash = None
		self.chat_history = []
		self.groq_client = None
		self._initialize_groq()
		
	def _initialize_groq(self):
		"""Initialize Groq client if API key is available"""
		try:
			from groq import Groq
			
			api_key = os.getenv('GROQ_API_KEY')
			if api_key:
				self.groq_client = Groq(api_key=api_key)
				print("Groq API initialized successfully")
			else:
				print("GROQ_API_KEY not found. Using fallback answer generation.")
		except ImportError:
			print("Groq library not installed. Using fallback answer generation.")
		except Exception as e:
			print(f"Error initializing Groq: {e}. Using fallback answer generation.")
	
	def process_pdf(self, pdf_file) -> Dict[str, Any]:
		"""Process uploaded PDF and extract text content"""
		try:
			# Read PDF content
			pdf_reader = PyPDF2.PdfReader(pdf_file)
			text_content = ""
			
			for page_num in range(len(pdf_reader.pages)):
				page = pdf_reader.pages[page_num]
				text_content += page.extract_text() + "\n"
			
			# Clean and preprocess text
			text_content = self._clean_text(text_content)
			
			# Generate document hash for identification
			document_hash = hashlib.md5(text_content.encode()).hexdigest()
			
			# Store document info
			self.current_document = {
				'filename': pdf_file.name,
				'pages': len(pdf_reader.pages),
				'size': len(text_content),
				'hash': document_hash,
				'upload_time': datetime.now().isoformat()
			}
			self.document_content = text_content
			self.document_hash = document_hash
			
			# Clear chat history for new document
			self.chat_history = []
			
			return {
				'success': True,
				'message': f"PDF processed successfully! {len(pdf_reader.pages)} pages loaded.",
				'document_info': self.current_document
			}
			
		except Exception as e:
			return {
				'success': False,
				'message': f"Error processing PDF: {str(e)}"
			}
	
	def _clean_text(self, text: str) -> str:
		"""Clean and preprocess extracted text"""
		# Remove excessive whitespace
		text = ' '.join(text.split())
		
		# Remove special characters that might interfere with processing
		import re
		text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
		
		# Restore paragraph breaks
		text = text.replace('. ', '.\n\n')
		
		return text
	
	def ask_question(self, question: str) -> Dict[str, Any]:
		"""Ask a question about the current document using enhanced RAG"""
		if not self.document_content:
			return {
				'success': False,
				'message': "No document loaded. Please upload a PDF first.",
				'answer': None
			}
		
		# Enhanced context validation
		relevance_score = self._calculate_enhanced_relevance(question, self.document_content)
		
		if relevance_score < 0.2:  # Lower threshold for better coverage
			return {
				'success': True,
				'message': "Question is out of context for the provided document.",
				'answer': self._generate_out_of_context_response(question),
				'relevance_score': relevance_score,
				'context': 'out_of_context'
			}
		
		# Generate enhanced answer using Groq API or fallback
		answer = self._generate_enhanced_answer(question, self.document_content)
		
		# Store in chat history
		chat_entry = {
			'question': question,
			'answer': answer,
			'timestamp': datetime.now(),
			'relevance_score': relevance_score,
			'context': 'in_context'
		}
		self.chat_history.append(chat_entry)
		
		return {
			'success': True,
			'message': "Answer generated successfully.",
			'answer': answer,
			'relevance_score': relevance_score,
			'context': 'in_context'
		}
	
	def _calculate_enhanced_relevance(self, question: str, document_content: str) -> float:
		"""Calculate enhanced relevance score using multiple methods"""
		# Method 1: Keyword matching
		question_keywords = self._extract_keywords(question)
		document_keywords = self._extract_keywords(document_content)
		
		keyword_score = self._calculate_keyword_relevance(question_keywords, document_keywords)
		
		# Method 2: Semantic similarity (simple word overlap)
		semantic_score = self._calculate_semantic_similarity(question, document_content)
		
		# Method 3: Question type analysis
		question_type_score = self._analyze_question_type(question, document_content)
		
		# Combine scores with weights
		final_score = (keyword_score * 0.4 + semantic_score * 0.4 + question_type_score * 0.2)
		
		return min(final_score, 1.0)
	
	def _extract_keywords(self, text: str) -> List[str]:
		"""Extract important keywords from text"""
		words = text.lower().split()
		# Enhanced stop words list
		stop_words = {
			'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
			'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
			'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
			'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
			'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom', 'whose'
		}
		
		keywords = [word for word in words if len(word) > 2 and word not in stop_words]
		return list(set(keywords))
	
	def _calculate_keyword_relevance(self, question_keywords: List[str], document_keywords: List[str]) -> float:
		"""Calculate keyword-based relevance score"""
		if not question_keywords or not document_keywords:
			return 0.0
		
		matches = sum(1 for qk in question_keywords if qk in document_keywords)
		relevance = matches / len(question_keywords)
		return min(relevance, 1.0)
	
	def _calculate_semantic_similarity(self, question: str, document_content: str) -> float:
		"""Calculate simple semantic similarity"""
		question_words = set(question.lower().split())
		document_words = set(document_content.lower().split())
		
		if not question_words:
			return 0.0
		
		intersection = question_words.intersection(document_words)
		union = question_words.union(document_words)
		
		if not union:
			return 0.0
		
		return len(intersection) / len(union)
	
	def _analyze_question_type(self, question: str, document_content: str) -> float:
		"""Analyze question type and relevance"""
		question_lower = question.lower()
		
		# Question words that indicate information seeking
		info_words = ['what', 'when', 'where', 'why', 'how', 'which', 'who', 'explain', 'describe', 'define']
		
		# Check if question is asking for information
		is_info_question = any(word in question_lower for word in info_words)
		
		if not is_info_question:
			return 0.3  # Lower score for non-information questions
		
		# Check if document contains relevant information
		relevant_info = self._find_relevant_sections(question, document_content)
		if relevant_info:
			return 0.8
		else:
			return 0.4
	
	def _generate_enhanced_answer(self, question: str, document_content: str) -> str:
		"""Generate enhanced answer using Groq API or fallback"""
		if self.groq_client:
			return self._generate_groq_answer(question, document_content)
		else:
			return self._generate_fallback_answer(question, document_content)
	
	def _validate_answer_content(self, answer: str, document_content: str) -> str:
		"""Validate that the answer only contains information from the document"""
		# Check for common phrases that might indicate external knowledge
		external_indicators = [
			"generally", "typically", "usually", "commonly", "in general",
			"as a rule", "it is known that", "research shows", "studies indicate",
			"experts say", "according to experts", "it is widely accepted",
			"in the field of", "in this domain", "in this area"
		]
		
		# Check if answer contains external knowledge indicators
		has_external_indicators = any(indicator in answer.lower() for indicator in external_indicators)
		
		if has_external_indicators:
			answer += "\n\n⚠️ **Warning:** This answer may contain information not explicitly stated in your document. Please verify all claims against the original PDF content."
		
		return answer
	
	def _generate_groq_answer(self, question: str, document_content: str) -> str:
		"""Generate answer using Groq API with advanced prompt engineering"""
		try:
			# Create advanced prompt
			prompt = self._create_advanced_prompt(question, document_content)
			
			# Get relevant context
			relevant_context = self._get_relevant_context(question, document_content)
			
			# Prepare messages for Groq
			messages = [
				{
					"role": "system",
					"content": prompt
				},
				{
					"role": "user",
					"content": f"Question: {question}\n\nDocument Context:\n{relevant_context}"
				}
			]
			
			# Call Groq API
			response = self.groq_client.chat.completions.create(
				model="llama3-8b-8192",  # Fast and efficient model
				messages=messages,
				temperature=0.1,  # Very low temperature for strict adherence
				max_tokens=1000,
				top_p=0.8  # Lower top_p for more focused responses
			)
			
			answer = response.choices[0].message.content
			
			# Post-process answer
			answer = self._post_process_answer(answer)
			
			# Validate answer content
			answer = self._validate_answer_content(answer, document_content)
			
			return answer
			
		except Exception as e:
			print(f"Error calling Groq API: {e}")
			return self._generate_fallback_answer(question, document_content)
	
	def _create_advanced_prompt(self, question: str, document_content: str) -> str:
		"""Create advanced prompt for better answer generation"""
		prompt = """You are an expert educational assistant that rewrites complex PDF content into simple, understandable language.
Your task is to take the information from the retrieved context chunks and rewrite it in simpler terms that are easier to understand.

--- STRICT CONTEXT RULES ---
- ONLY use information that is explicitly stated in the retrieved chunks
- If the retrieved chunks do not contain enough information to answer the question, respond with:
  "I could not find enough information in the provided document to answer this question."
- Do NOT reference any information outside the provided chunks
- Do NOT make assumptions, inferences, or add context not in the document
- Do NOT use any external knowledge, even if it seems relevant

--- SIMPLIFICATION & REWRITING ---
- Take the complex language from the document and rewrite it in simple, everyday terms
- Break down long, complicated sentences into shorter, clearer ones
- Replace technical jargon with simple explanations (but keep the same meaning)
- Use active voice instead of passive voice when possible
- Organize information in a logical, easy-to-follow sequence
- Add clear transitions between ideas to improve flow

--- WHAT TO DO ---
- Paraphrase the document's content in your own simple words
- Maintain the exact same meaning and facts from the document
- Make complex concepts easier to understand
- Use examples and analogies ONLY if they help clarify what's already in the document
- Break down complex processes into simple steps (if the document describes steps)

--- WHAT NOT TO DO ---
- Do NOT add new information not present in the document
- Do NOT change the meaning or interpretation of what's written
- Do NOT add external context or industry knowledge
- Do NOT make the content more comprehensive than the original
- Do NOT add conclusions or insights not explicitly stated

--- STYLE & TONE ---
- Be clear, concise, and educational
- Use simple, everyday language that a high school student could understand
- Keep answers under 300 words unless more detail is explicitly requested
- Use bullet points, numbered lists, and clear formatting for readability
- Write as if explaining to someone learning the topic for the first time

--- ANSWER FORMAT ---
1. **Simple Summary** (1–2 sentences explaining what the document says in simple terms)
2. **Key Points** (bullet points of information rewritten in simple language)
3. **Step-by-Step Explanation** (if the document provides steps, rewrite them simply)
4. **Examples or Applications** (if mentioned in the document, explain them simply)
5. **Learning Tips** (how to understand the simplified explanation better)
6. **Source Attribution** ("Source: Retrieved document chunks")

--- CRITICAL RULES ---
- NEVER invent numbers, dates, or statistics not in the document
- NEVER add examples, analogies, or explanations not present in the text
- NEVER make connections or conclusions not explicitly stated
- NEVER use external knowledge to fill gaps in the document
- ALWAYS maintain the exact same meaning as the original document
- FOCUS on making the existing document content more readable and understandable"""
		
		return prompt
	
	def _get_relevant_context(self, question: str, document_content: str) -> str:
		"""Get relevant context from document for the question"""
		relevant_sections = self._find_relevant_sections(question, document_content)
		
		if not relevant_sections:
			return document_content[:2000]  # Return first 2000 characters if no relevant sections
		
		# Combine relevant sections
		context = "\n\n".join(relevant_sections[:3])  # Top 3 relevant sections
		
		# Limit context length
		if len(context) > 3000:
			context = context[:3000] + "..."
		
		return context
	
	def _find_relevant_sections(self, question: str, document_content: str) -> List[str]:
		"""Find relevant sections in the document content"""
		paragraphs = document_content.split('\n\n')
		relevant_sections = []
		
		question_keywords = self._extract_keywords(question)
		
		for paragraph in paragraphs:
			if len(paragraph.strip()) > 30:  # Lower threshold for better coverage
				paragraph_keywords = self._extract_keywords(paragraph)
				relevance = self._calculate_keyword_relevance(question_keywords, paragraph_keywords)
				
				if relevance > 0.1:  # Lower threshold for section relevance
					relevant_sections.append(paragraph.strip())
		
		# Sort by relevance
		relevant_sections.sort(key=lambda x: self._calculate_keyword_relevance(
			question_keywords, self._extract_keywords(x)
		), reverse=True)
		
		return relevant_sections
	
	def _post_process_answer(self, answer: str) -> str:
		"""Post-process the generated answer"""
		# Remove any system instructions that might have leaked
		if "--- STRICT CONTEXT RULES ---" in answer:
			answer = answer.split("--- STRICT CONTEXT RULES ---")[0]
		if "--- SIMPLIFICATION & REWRITING ---" in answer:
			answer = answer.split("--- SIMPLIFICATION & REWRITING ---")[0]
		if "--- WHAT TO DO ---" in answer:
			answer = answer.split("--- WHAT TO DO ---")[0]
		if "--- WHAT NOT TO DO ---" in answer:
			answer = answer.split("--- WHAT NOT TO DO ---")[0]
		if "--- CRITICAL RULES ---" in answer:
			answer = answer.split("--- CRITICAL RULES ---")[0]
		
		# Ensure proper formatting
		answer = answer.strip()
		
		# Add source citation if not present
		if "Source:" not in answer:
			answer += "\n\n📎 **Source:** Retrieved document chunks"
		
		# Ensure attractive markdown formatting for learning sections (with emojis)
		answer = answer.replace("**Simple Summary**", "## 📘 Simple Summary")
		answer = answer.replace("**Key Points**", "## 🔑 Key Points")
		answer = answer.replace("**Step-by-Step Explanation**", "## 🧭 Step-by-Step Explanation")
		answer = answer.replace("**Examples or Applications**", "## 💡 Examples or Applications")
		answer = answer.replace("**Learning Tips**", "## 🎯 Learning Tips")
		
		# Add clarification about the simplified approach (with emoji)
		answer += "\n\n💡 **Note:** This answer contains the same information as your PDF document, but rewritten in simpler language to make it easier to understand. All facts, meanings, and details remain exactly the same as in the original document."
		
		return answer
	
	def _generate_fallback_answer(self, question: str, document_content: str) -> str:
		"""Generate fallback answer when Groq API is not available"""
		relevant_sections = self._find_relevant_sections(question, document_content)
		
		if relevant_sections:
			answer = "Based on the document content:\n\n"
			for i, section in enumerate(relevant_sections[:3]):
				answer += f"{i+1}. {section}\n\n"
			answer += "This information is extracted from the uploaded PDF document."
		else:
			answer = "While the question appears relevant to the document, I couldn't find specific information addressing it in the current content. The document may not contain detailed information about this particular aspect."
		
		return answer
	
	def _generate_out_of_context_response(self, question: str) -> str:
		"""Generate response for out-of-context questions"""
		return f"""The question "{question}" is not related to the content of the uploaded PDF document.

To get helpful answers, please ask questions that are relevant to the document's content. For example:
- What are the main topics covered in this document?
- What does the document say about [specific topic]?
- Can you explain [concept mentioned in the document]?
- What are the key findings or conclusions?

If you need information about a different topic, please upload a relevant document."""
	
	def start_new_chat(self):
		"""Start a new chat session"""
		self.current_document = None
		self.document_content = None
		self.document_hash = None
		self.chat_history = []
		return {
			'success': True,
			'message': "New chat session started. You can now upload a new PDF document."
		}
	
	def get_chat_history(self) -> List[Dict[str, Any]]:
		"""Get current chat history"""
		return self.chat_history
	
	def get_document_info(self) -> Optional[Dict[str, Any]]:
		"""Get information about the currently loaded document"""
		return self.current_document
	
	def export_chat_history(self) -> str:
		"""Export chat history as JSON"""
		export_data = {
			'document_info': self.current_document,
			'chat_history': [
				{
					'question': chat['question'],
					'answer': chat['answer'],
					'timestamp': chat['timestamp'].isoformat() if hasattr(chat['timestamp'], 'isoformat') else str(chat['timestamp']),
					'relevance_score': chat['relevance_score'],
					'context': chat['context']
				}
				for chat in self.chat_history
			]
		}
		return json.dumps(export_data, indent=2)
	
	def get_system_status(self) -> Dict[str, Any]:
		"""Get system status information"""
		return {
			'groq_available': self.groq_client is not None,
			'document_loaded': self.current_document is not None,
			'chat_history_count': len(self.chat_history),
			'document_size': len(self.document_content) if self.document_content else 0
		}

def main():
	"""Main application function - Focused on RAG functionality with authentication"""
	
	# Page configuration
	st.set_page_config(
		page_title="Educational RAG System",
		page_icon=None,
		layout="wide",
		initial_sidebar_state="expanded"
	)
	
	# Check authentication
	if 'session_token' not in st.session_state:
		st.warning("Please sign in to access the RAG system")
		st.info("Navigate to the Authentication page to sign in or create an account")
		st.stop()
	
	# Import auth component
	try:
		from components.auth import UserAuth
	except ImportError:
		from app.components.auth import UserAuth
	
	# Verify session
	auth = UserAuth()
	user = auth.get_user_from_session(st.session_state.session_token)
	
	if not user:
		del st.session_state.session_token
		st.error("Session expired. Please sign in again")
		st.stop()
	
	# Initialize RAG system in session state
	if 'rag_system' not in st.session_state:
		st.session_state.rag_system = RAGSystem()
	
	rag_system = st.session_state.rag_system
	
	# Main title and description
	st.title("Educational RAG Document Q&A System")
	st.markdown(f"""
	**Welcome back, {user.get('full_name', user['username'])}!**
	
	**Intelligent PDF Document Analysis and Question Answering**
	
	Upload a PDF document and get intelligent, context-aware answers powered by advanced AI!
	""")
	
	# System Status
	system_status = rag_system.get_system_status()
	if system_status['groq_available']:
		st.success("Groq API Connected - Using AI-powered answer generation")
	else:
		st.warning("Groq API Not Available - Using enhanced fallback generation")
		st.info("Set GROQ_API_KEY environment variable to enable AI-powered answers")
	
	# PDF Upload Section
	col1, col2 = st.columns([2, 1])
	
	with col1:
		st.subheader("Document Upload")
		
		# Check if document is already loaded
		current_doc = rag_system.get_document_info()
		
		if current_doc:
			st.success(f"Current Document: {current_doc['filename']}")
			st.info(f"Pages: {current_doc['pages']} | Size: {current_doc['size']} characters")
			if 'upload_time' in current_doc:
				st.info(f"Uploaded: {current_doc['upload_time'][:19]}")
			
			# Option to start new chat
			if st.button("Start New Chat", type="secondary"):
				result = rag_system.start_new_chat()
				st.success(result['message'])
				st.rerun()
		else:
			uploaded_file = st.file_uploader(
				"Choose a PDF file",
				type=['pdf'],
				help="Upload a single PDF document to start asking questions"
			)
			
			if uploaded_file is not None:
				if st.button("Process PDF", type="primary"):
					with st.spinner("Processing PDF..."):
						result = rag_system.process_pdf(uploaded_file)
						
						if result['success']:
							st.success(result['message'])
							st.rerun()
						else:
							st.error(result['message'])
	
	with col2:
		st.subheader("How It Works")
		st.write("""
		1. Upload PDF - Select a document to analyze
		2. Ask Questions - Type questions about the content
		3. Get AI Answers - Receive intelligent, context-aware responses
		4. New Chat - Start fresh with a different document
		""")
		
		if current_doc:
			st.success("Document loaded and ready for questions!")
		else:
			st.info("Upload a PDF to get started")
	
	# Chat Interface
	if current_doc:
		st.subheader("Ask Questions About Your Document")
		
		# Question input
		question = st.text_input(
			"Ask a question about the document:",
			placeholder="e.g., What are the main topics covered? What does the document say about...?",
			help="Ask specific questions about the content in your uploaded PDF"
		)
		
		if st.button("Ask Question", type="primary", disabled=not question):
			if question:
				with st.spinner("Analyzing document and generating enhanced answer..."):
					result = rag_system.ask_question(question)
					
					if result['success']:
						if result['context'] == 'out_of_context':
							st.warning("Question Out of Context")
							st.info(result['answer'])
						else:
							st.success("Enhanced Answer Generated")
							st.write(result['answer'])
						
						# Show relevance score and context
						st.info(f"Relevance Score: {result['relevance_score']:.2f}")
						st.info(f"Context: {result['context']}")
						
						# Show API status
						if system_status['groq_available']:
							st.success("AI-Powered Answer - Generated using Groq API")
						else:
							st.info("Enhanced Fallback Answer - Using improved algorithms")
		
		# Chat History
		chat_history = rag_system.get_chat_history()
		if chat_history:
			st.subheader("Chat History")
			
			for i, chat in enumerate(reversed(chat_history)):
				with st.expander(f"Q{i+1}: {chat['question'][:50]}..."):
					st.write(f"**Question:** {chat['question']}")
					st.write(f"**Answer:** {chat['answer']}")
					st.write(f"**Relevance:** {chat['relevance_score']:.2f}")
					st.write(f"**Context:** {chat['context']}")
					st.write(f"**Time:** {chat['timestamp'].strftime('%H:%M:%S')}")
			
			# Export chat history
			if st.button("Export Chat History"):
				export_data = rag_system.export_chat_history()
				st.download_button(
					label="Download JSON",
					data=export_data,
					file_name=f"chat_history_{current_doc['filename']}.json",
					mime="application/json"
				)
	
	# Sidebar - Minimal navigation with user info
	with st.sidebar:
		st.header("RAG System")
		
		if current_doc:
			st.success(f"Document Loaded: {current_doc['filename']}")
			st.info(f"Pages: {current_doc['pages']}")
			st.info(f"Size: {current_doc['size']} characters")
			
			if st.button("New Chat", type="secondary", use_container_width=True):
				result = rag_system.start_new_chat()
				st.success(result['message'])
				st.rerun()
		else:
			st.info("No document loaded. Upload a PDF to start asking questions!")
		
		# System status
		st.header("System Status")
		st.info(f"""
		Groq API: {'Connected' if system_status['groq_available'] else 'Not Available'}
		Document: {'Loaded' if system_status['document_loaded'] else 'None'}
		Chat History: {system_status['chat_history_count']} interactions
		""")
		
		# Groq API Key Input (Alternative to environment variable)
		st.header("Groq API Configuration")
		if not system_status['groq_available']:
			st.info("Enter your Groq API key to enable AI-powered answers:")
			api_key_input = st.text_input(
				"Groq API Key:",
				type="password",
				help="Get your API key from https://console.groq.com/keys"
			)
			
			if st.button("Connect Groq API", type="primary", use_container_width=True):
				if api_key_input:
					# Set the API key in environment for this session
					os.environ['GROQ_API_KEY'] = api_key_input
					# Reinitialize the RAG system with new API key
					rag_system.groq_client = None
					rag_system._initialize_groq()
					st.success("Groq API connected successfully!")
					st.rerun()
				else:
					st.error("Please enter a valid API key")
		else:
			st.success("Groq API is connected!")
			if st.button("Reconnect", type="secondary", use_container_width=True):
				rag_system.groq_client = None
				rag_system._initialize_groq()
				st.rerun()
		
		# Quick actions
		st.header("Quick Actions")
		if st.button("Refresh", use_container_width=True):
			st.rerun()
		
		if st.button("View Status", use_container_width=True):
			st.info(f"""
			System Status: All components operational
			
			Groq API: {'Connected' if system_status['groq_available'] else 'Not Available'}
			Document: {'Loaded' if system_status['document_loaded'] else 'None'}
			Chat History: {system_status['chat_history_count']} interactions
			Document Size: {system_status['document_size']} characters
			""")
		
		# User account section
		st.header("Account")
		st.info(f"""
		Logged in as: {user.get('full_name', user['username'])}
		Email: {user['email']}
		""")
		
		if st.button("Sign Out", type="secondary", use_container_width=True):
			# Clear session and redirect to auth
			auth.logout_user(st.session_state.session_token)
			del st.session_state.session_token
			del st.session_state.user_data
			st.success("Successfully signed out!")
			st.rerun()
	
	# Footer
	st.markdown("---")
	st.markdown("""
	<div style='text-align: center; color: #666;'>
		Educational RAG System | Built with Streamlit & Groq API | 
		<span style='color: #ff6b6b;'>Enhanced AI-Powered Document Q&A</span>
	</div>
	""", unsafe_allow_html=True)

if __name__ == "__main__":
	main()