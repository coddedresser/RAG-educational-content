# Step-by-Step Implementation Guide

## Replace Current RAG System with Enhanced Version

### Step 1: Install Groq Library

```bash
pip install groq
```

### Step 2: Get Groq API Key

1. Go to [https://console.groq.com/](https://console.groq.com/)
2. Sign up for free account
3. Create API key
4. Set environment variable: `GROQ_API_KEY=your_key_here`

### Step 3: Replace RAGSystem Class in main.py

**Find this section in your main.py (around line 52):**

```python
class RAGSystem:
    """RAG (Retrieval Augmented Generation) system for PDF document processing and Q&A"""
    
    def __init__(self):
        self.current_document = None
        self.document_content = None 
        self.document_hash = None
        self.chat_history = []
```

**Replace the ENTIRE RAGSystem class with this enhanced version:**

```python
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
            import os
            from groq import Groq
            
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                self.groq_client = Groq(api_key=api_key)
                print("✅ Groq API initialized successfully")
            else:
                print("⚠️ GROQ_API_KEY not found. Using fallback answer generation.")
        except ImportError:
            print("⚠️ Groq library not installed. Using fallback answer generation.")
        except Exception as e:
            print(f"⚠️ Error initializing Groq: {e}. Using fallback answer generation.")
    
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
            'context': 'in_context',
            'relevance_score': relevance_score
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
                temperature=0.3,  # Lower temperature for more focused answers
                max_tokens=1000,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            
            # Post-process answer
            answer = self._post_process_answer(answer)
            
            return answer
            
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return self._generate_fallback_answer(question, document_content)
    
    def _create_advanced_prompt(self, question: str, document_content: str) -> str:
        """Create advanced prompt for better answer generation"""
        prompt = """You are an expert document analysis assistant. Your task is to provide clear, accurate, and comprehensive answers based on the provided document content.

IMPORTANT GUIDELINES:
1. Answer ONLY based on the information provided in the document
2. If the document doesn't contain enough information, clearly state this
3. Provide specific examples and quotes from the document when possible
4. Structure your answer logically with clear sections
5. Use bullet points for lists and key information
6. Be concise but thorough
7. If the question is ambiguous, ask for clarification
8. Always cite the source as "Based on the document content"

ANSWER FORMAT:
- Start with a direct answer to the question
- Provide supporting details from the document
- Include relevant examples or quotes
- Summarize key points
- End with "Source: Uploaded PDF document"

Remember: Accuracy and relevance are more important than length."""
        
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
        if "IMPORTANT GUIDELINES:" in answer:
            answer = answer.split("IMPORTANT GUIDELINES:")[0]
        
        # Ensure proper formatting
        answer = answer.strip()
        
        # Add source citation if not present
        if "Source:" not in answer:
            answer += "\n\nSource: Uploaded PDF document"
        
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
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        return {
            'groq_available': self.groq_client is not None,
            'document_loaded': self.current_document is not None,
            'chat_history_count': len(self.chat_history),
            'document_size': len(self.document_content) if self.document_content else 0
        }
```

### Step 4: Add Import for datetime

**Add this import at the top of your main.py:**

```python
from datetime import datetime
```

### Step 5: Update Chat History Display

**Find this section in your main.py:**

```python
# Chat History
chat_history = rag_system.get_chat_history()
if chat_history:
    st.subheader("📝 Chat History")
    
    for i, chat in enumerate(reversed(chat_history)):
        with st.expander(f"Q{i+1}: {chat['question'][:50]}..."):
            st.write(f"**Question:** {chat['question']}")
            st.write(f"**Answer:** {chat['answer']}")
            st.write(f"**Relevance:** {chat['relevance_score']:.2f}")
            st.write(f"**Time:** {chat['timestamp'].strftime('%H:%M:%S')}")
```

**Replace it with:**

```python
# Chat History
chat_history = rag_system.get_chat_history()
if chat_history:
    st.subheader("📝 Enhanced Chat History")
    
    for i, chat in enumerate(reversed(chat_history)):
        with st.expander(f"Q{i+1}: {chat['question'][:50]}..."):
            st.write(f"**Question:** {chat['question']}")
            st.write(f"**Answer:** {chat['answer']}")
            st.write(f"**Relevance:** {chat['relevance_score']:.2f}")
            st.write(f"**Context:** {chat['context']}")
            st.write(f"**Time:** {chat['timestamp'].strftime('%H:%M:%S')}")
    
    # Export chat history
    if st.button("📤 Export Chat History"):
        import json
        export_data = {
            'document_info': rag_system.get_document_info(),
            'chat_history': [
                {
                    'question': chat['question'],
                    'answer': chat['answer'],
                    'timestamp': chat['timestamp'].isoformat() if hasattr(chat['timestamp'], 'isoformat') else str(chat['timestamp']),
                    'relevance_score': chat['relevance_score'],
                    'context': chat['context']
                }
                for chat in chat_history
            ]
        }
        st.download_button(
            label="Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"chat_history_{rag_system.get_document_info()['filename']}.json",
            mime="application/json"
        )
```

### Step 6: Add System Status Display

**Add this section after the RAG header:**

```python
# System Status
system_status = rag_system.get_system_status()
if system_status['groq_available']:
    st.success("✅ **Groq API Connected** - Using AI-powered answer generation")
else:
    st.warning("⚠️ **Groq API Not Available** - Using enhanced fallback generation")
    st.info("Set GROQ_API_KEY environment variable to enable AI-powered answers")
```

### Step 7: Test the Enhanced System

1. **Set your Groq API key:**
   ```bash
   set GROQ_API_KEY=your_key_here  # Windows
   export GROQ_API_KEY=your_key_here  # Linux/Mac
   ```

2. **Run the application:**
   ```bash
   streamlit run app/main.py
   ```

3. **Upload a PDF and ask questions**

## What You'll Get

✅ **Better Answers**: AI-powered responses using Groq API  
✅ **Improved Context Detection**: Multi-method relevance scoring  
✅ **Enhanced User Experience**: Export functionality, better status display  
✅ **Fallback System**: Works even without API key  
✅ **Advanced Prompt Engineering**: Optimized for clear, accurate answers  

## Troubleshooting

- **API Key Issues**: Check environment variable is set correctly
- **Import Errors**: Ensure `pip install groq` completed successfully
- **Performance**: Monitor Groq API usage in console.groq.com

The enhanced system will provide significantly better answers and a more intelligent question-answering experience!
