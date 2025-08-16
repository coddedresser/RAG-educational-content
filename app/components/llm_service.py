"""
Free LLM Service using Hugging Face Inference API
"""
import requests
import json
import logging
from typing import Dict, List, Any, Optional
import time

class FreeLLMService:
    """Free LLM service using Hugging Face Inference API"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://api-inference.huggingface.co/models"
        
        # Free models that work well for education
        self.models = {
            "text_generation": "microsoft/DialoGPT-medium",  # Conversational
            "summarization": "facebook/bart-large-cnn",      # Summarization
            "question_answering": "deepset/roberta-base-squad2",  # Q&A
            "text_classification": "distilbert-base-uncased-finetuned-sst-2-english"  # Classification
        }
        
        # No API key required for basic usage (rate limited)
        self.headers = {"Content-Type": "application/json"}
    
    def generate_learning_path(self, user_query: str, student_profile: Dict, available_content: List) -> Dict[str, Any]:
        """Generate AI-powered learning path using free LLM"""
        try:
            # Create prompt for learning path generation
            prompt = self._create_learning_path_prompt(user_query, student_profile, available_content)
            
            # Use text generation model
            response = self._call_huggingface_api(
                model=self.models["text_generation"],
                inputs=prompt,
                task="text-generation"
            )
            
            if response:
                # Parse and structure the response
                learning_path = self._parse_learning_path_response(response, available_content)
                return learning_path
            else:
                return self._fallback_learning_path(user_query, student_profile, available_content)
                
        except Exception as e:
            self.logger.error(f"Error generating learning path: {e}")
            return self._fallback_learning_path(user_query, student_profile, available_content)
    
    def answer_educational_question(self, question: str, context: str) -> str:
        """Answer educational questions using free Q&A model"""
        try:
            # Use question-answering model
            response = self._call_huggingface_api(
                model=self.models["question_answering"],
                inputs={
                    "question": question,
                    "context": context
                },
                task="question-answering"
            )
            
            if response and isinstance(response, dict):
                return response.get('answer', 'I cannot answer that question at the moment.')
            else:
                return 'I cannot answer that question at the moment.'
                
        except Exception as e:
            self.logger.error(f"Error answering question: {e}")
            return 'I cannot answer that question at the moment.'
    
    def generate_content_summary(self, content: str) -> str:
        """Generate content summaries using free summarization model"""
        try:
            # Use summarization model
            response = self._call_huggingface_api(
                model=self.models["summarization"],
                inputs=content[:1000],  # Limit input length
                task="summarization"
            )
            
            if response and isinstance(response, list) and len(response) > 0:
                return response[0].get('summary_text', content[:200] + '...')
            else:
                return content[:200] + '...'
                
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return content[:200] + '...'
    
    def classify_learning_style(self, student_responses: List[str]) -> str:
        """Classify learning style using free classification model"""
        try:
            # Combine responses for classification
            combined_text = " ".join(student_responses)
            
            response = self._call_huggingface_api(
                model=self.models["text_classification"],
                inputs=combined_text,
                task="text-classification"
            )
            
            if response and isinstance(response, list) and len(response) > 0:
                # Map classification to learning style
                return self._map_classification_to_learning_style(response[0])
            else:
                return 'visual'  # Default fallback
                
        except Exception as e:
            self.logger.error(f"Error classifying learning style: {e}")
            return 'visual'  # Default fallback
    
    def _call_huggingface_api(self, model: str, inputs: Any, task: str) -> Any:
        """Call Hugging Face Inference API"""
        try:
            url = f"{self.base_url}/{model}"
            
            payload = {
                "inputs": inputs,
                "options": {
                    "wait_for_model": True,
                    "use_cache": False
                }
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                # Model is loading, wait and retry
                time.sleep(10)
                response = requests.post(url, headers=self.headers, json=payload, timeout=30)
                if response.status_code == 200:
                    return response.json()
                else:
                    self.logger.warning(f"Model {model} not available: {response.status_code}")
                    return None
            else:
                self.logger.warning(f"API call failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error calling Hugging Face API: {e}")
            return None
    
    def _create_learning_path_prompt(self, user_query: str, student_profile: Dict, available_content: List) -> str:
        """Create prompt for learning path generation"""
        content_summary = "\n".join([f"- {c['title']} ({c['subject']}, {c['difficulty_level']})" 
                                   for c in available_content[:10]])  # Limit to first 10
        
        prompt = f"""
        Create a personalized learning path for an educational query.
        
        User Query: {user_query}
        Student Profile: {json.dumps(student_profile, indent=2)}
        
        Available Content:
        {content_summary}
        
        Generate a structured learning path with:
        1. Learning objectives
        2. Recommended content sequence
        3. Estimated time for each item
        4. Difficulty progression
        5. Success criteria
        
        Format the response as a clear, structured learning path.
        """
        
        return prompt
    
    def _parse_learning_path_response(self, response: Any, available_content: List) -> Dict[str, Any]:
        """Parse LLM response into structured learning path"""
        try:
            if isinstance(response, list) and len(response) > 0:
                generated_text = response[0].get('generated_text', '')
            else:
                generated_text = str(response)
            
            # Extract key information from generated text
            learning_path = {
                'title': f"AI-Generated Learning Path",
                'objectives': self._extract_objectives(generated_text),
                'content_sequence': self._match_content_to_sequence(generated_text, available_content),
                'estimated_time': self._calculate_total_time(available_content),
                'difficulty_progression': self._extract_difficulty_progression(generated_text),
                'success_criteria': self._extract_success_criteria(generated_text),
                'ai_generated': True
            }
            
            return learning_path
            
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            return self._fallback_learning_path("", {}, available_content)
    
    def _fallback_learning_path(self, user_query: str, student_profile: Dict, available_content: List) -> Dict[str, Any]:
        """Fallback learning path when LLM is unavailable"""
        return {
            'title': f"Learning Path for {user_query or 'Your Goals'}",
            'objectives': ["Complete the selected content", "Achieve understanding of key concepts"],
            'content_sequence': available_content[:5],  # First 5 items
            'estimated_time': sum(c.get('estimated_time', 0) for c in available_content[:5]),
            'difficulty_progression': [c['difficulty_level'] for c in available_content[:5]],
            'success_criteria': ["Complete all items", "Score above 80% on assessments"],
            'ai_generated': False
        }
    
    def _extract_objectives(self, text: str) -> List[str]:
        """Extract learning objectives from generated text"""
        # Simple extraction - you can make this more sophisticated
        objectives = []
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['objective', 'goal', 'learn', 'understand']):
                objectives.append(line.strip())
        
        return objectives[:5] if objectives else ["Complete the learning path successfully"]
    
    def _match_content_to_sequence(self, text: str, available_content: List) -> List[Dict]:
        """Match available content to AI-generated sequence"""
        # Simple matching - you can make this more sophisticated
        return available_content[:5]  # Return first 5 items for now
    
    def _calculate_total_time(self, content_sequence: List[Dict]) -> int:
        """Calculate total estimated time for learning path"""
        return sum(c.get('estimated_time', 0) for c in content_sequence)
    
    def _extract_difficulty_progression(self, text: str) -> List[str]:
        """Extract difficulty progression from generated text"""
        # Simple extraction
        difficulties = ['beginner', 'intermediate', 'advanced']
        found = []
        for diff in difficulties:
            if diff in text.lower():
                found.append(diff)
        
        return found if found else ['beginner', 'intermediate']
    
    def _extract_success_criteria(self, text: str) -> List[str]:
        """Extract success criteria from generated text"""
        # Simple extraction
        criteria = []
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['success', 'complete', 'achieve', 'score']):
                criteria.append(line.strip())
        
        return criteria[:3] if criteria else ["Complete all learning objectives"]
    
    def _map_classification_to_learning_style(self, classification: Dict) -> str:
        """Map classification result to learning style"""
        # Simple mapping - you can make this more sophisticated
        label = classification.get('label', '').lower()
        score = classification.get('score', 0)
        
        if 'visual' in label or score > 0.7:
            return 'visual'
        elif 'auditory' in label:
            return 'auditory'
        elif 'kinesthetic' in label:
            return 'kinesthetic'
        else:
            return 'reading'  # Default

# Example usage
if __name__ == "__main__":
    # Test the free LLM service
    llm_service = FreeLLMService()
    
    # Test question answering
    question = "What is algebra?"
    context = "Algebra is a branch of mathematics that deals with symbols and the rules for manipulating these symbols."
    
    answer = llm_service.answer_educational_question(question, context)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    # Test content summarization
    content = "Algebra is a fundamental branch of mathematics that introduces the concept of variables and equations. It provides tools for solving problems using symbols and mathematical operations."
    summary = llm_service.generate_content_summary(content)
    print(f"\nContent: {content}")
    print(f"Summary: {summary}")