import os
from dotenv import load_dotenv
import json
from openai import OpenAI
from prompt_templates import PromptTemplates
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress httpx INFO logs (HTTP request/response messages)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

class EvidenceExtractor:
    """
    Uses DeepSeek R1 to extract evidence from text chunks with predefined prompt templates.
    """
    
    def __init__(self, model: str = "deepseek-chat"):
        """Initialize the evidence extractor"""
        
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key = self.api_key, base_url = 'https://api.deepseek.com')
        self.model = model
        self.prompts = PromptTemplates()
        self.request_count = 0
    
    def extract_from_chunk(self, chunk_text: str, prompt_type: str) -> dict:
        """
        Extract evidence from a text chunk using a predefined prompt template
        
        Args:
            chunk_text: The text chunk to extract evidence from
            prompt_type: The type of prompt to use (ai_features, performance_degradation, causal_links)
        
        Returns:
            Dictionary containing the extracted evidence
        """
        
        prompt = self.prompts.EXTRACTION_PROMPTS[prompt_type].format(
            RESEARCH_CONTEXT=self.prompts.RESEARCH_CONTEXT,
            chunk_text=chunk_text
        )
                
        try: 
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a precision text extraction tool with ZERO-TOLERANCE for hallucination.

CORE PRINCIPLES:
- You ONLY extract text that exists VERBATIM in the provided chunk
- You NEVER create, invent, paraphrase, or modify any text  
- You NEVER use external knowledge to fill gaps
- You NEVER extract less than 10 words
- If no relevant text exists in the chunk, you return empty results

VERIFICATION REQUIRED:
Before including any excerpt, you must be able to point to its exact location in the source text. Every word must exist in the original chunk in the same order.

POSITION-BASED VERIFICATION:
For each excerpt, you must be able to specify:
- The approximate location in the chunk (beginning/middle/end)  
- The first 3-4 words that start the excerpt
- The last 3-4 words that end the excerpt
- Confirmation that these boundaries exist in the source

EXAMPLE VERIFICATION:
✅ "Located in middle section, starts with 'runtime learning/adaptation cannot', ends with 'within safe limits'"
❌ Cannot specify exact boundaries = HALLUCINATION

HALLUCINATION = FAILURE. Extract nothing rather than fabricate anything.

JSON FORMAT REQUIREMENTS:
Your response MUST be valid JSON with proper structure:
- Use objects {} not JSON strings for nested data  
- All numeric scores must be actual numbers, not strings
- Array elements must be objects, never strings or other types
- Ensure proper comma placement and bracket matching"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,  # Maximum determinism - no creativity allowed
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            self.request_count += 1
            
            # Parse the JSON response
            message = response.choices[0].message
            response_content = message.content
            
            logger.info(f"API response received: {response_content}")
            
            # Handle empty response
            if not response_content or response_content.strip() == "":
                logger.error("Empty response content from DeepSeek API")
                return {
                    "success": False,
                    "error": "Empty response from model",
                    "request_count": self.request_count,
                    "prompt_type": prompt_type,
                    "model": self.model
                }
            
            response_json = json.loads(response_content)
            
            # Validate JSON structure 
            if not isinstance(response_json, dict):
                logger.error(f"Response is not a dictionary: {type(response_json)}")
                return {
                    "success": False,
                    "error": f"Invalid response type: expected dict, got {type(response_json)}",
                    "request_count": self.request_count,
                    "prompt_type": prompt_type,
                    "model": self.model
                }
            
            # Check for expected top-level keys
            expected_keys = ['features', 'degradations', 'causal_links']
            if not any(key in response_json for key in expected_keys):
                logger.error(f"Response missing expected keys. Found: {list(response_json.keys())}")
                return {
                    "success": False,
                    "error": f"Missing expected keys. Found: {list(response_json.keys())}",
                    "request_count": self.request_count,
                    "prompt_type": prompt_type,
                    "model": self.model
                }
            
            # Validate that array values contain only dictionaries
            for key, value in response_json.items():
                if isinstance(value, list):
                    original_count = len(value)
                    valid_items = [x for x in value if isinstance(x, dict)]
                    if len(valid_items) < original_count:
                        invalid_count = original_count - len(valid_items)
                        logger.warning(f"Filtered {invalid_count} invalid items from {key} (expected dicts)")
                        response_json[key] = valid_items
            
            return {
                "success": True,
                "evidence": response_json,
                "request_count": self.request_count,
                "prompt_type": prompt_type,
                "model": self.model
            }
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Log the raw content for debugging
            raw_content = response_content if 'response_content' in dir() else 'N/A'
            logger.error(f"Raw response (first 500 chars): {raw_content[:500] if raw_content else 'Empty'}")
            return {
                "success": False,
                "error": f"JSON parsing error: {str(e)}",
                "request_count": self.request_count,
                "prompt_type": prompt_type,
                "model": self.model
            }
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "request_count": self.request_count,
                "prompt_type": prompt_type,
                "model": self.model
            }
    
    def extract_batch(self, chunks: List[Dict], prompt_type: str, max_workers: int = 3) -> List[Dict]:
        """
        Extract evidence from a list of text chunks using a predefined prompt template
        
        Args:
            chunks: List of text chunks to extract evidence from
            prompt_type: The type of prompt to use (ai_features, performance_degradation, causal_links)
            max_workers: Maximum number of workers to use for parallel extraction
        
        Returns:
            List of dictionaries containing the extracted evidence
        """
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.extract_from_chunk, chunk['text'], prompt_type): chunk
                for chunk in chunks
            }
            
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    result = future.result()
                    result['chunk_id'] = chunk['chunk_id']
                    result['document_id'] = chunk['document_id']
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error extracting evidence from chunk {chunk['chunk_id']}: {e}")
                    results.append({
                        "success": False,
                        "error": str(e),
                        "chunk_id": chunk['chunk_id'],
                        "document_id": chunk['document_id'],
                        "prompt_type": prompt_type,
                        "model": self.model
                    })
        
        return results