import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class DocumentDomainExtractor:
    """
    Extracts industry domains from document text
    """
    
    def __init__(self, model: str = "deepseek-chat"):
        """Initialize the domain extractor"""
        
        # Initialize DeepSeek API key
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key = self.api_key, base_url = 'https://api.deepseek.com')
        
        # Initialize model name
        self.model = model
    
    def extract_domains(self, document_text: str, metadata: dict) -> dict:
        """
        Uses DeepSeek LLM to identify any relevant industry domains in the document text
        
        Input: Abstract + introduction section of the document (most likely to contain the industry domains)
        Output: Dictionary with 'domains' key and list of industry domains as value
        
        Raises:
            ValueError: If API call fails or response is invalid
        """
        
        # Prompt Structure:
        prompt = f"""
        Analyze the abstract and introduction sections of this academic research paper. 
        Identify ALL VALID relevant HIGH-RISK INDUSTRY DOMAINS mentioned in the text.
        
        VALID INDUSTRY DOMAINS:
        - Nuclear Energy
        - Oil & Gas
        - Transportation
        - Maritime
        - Offshore
        - Construction
        - Aviation
        - Railway
        - Automotive
        - Manufacturing
        - Robotics
        - Healthcare
        - Power/Energy
        - Agriculture
        
        Document Title: {metadata.get('title', 'N/A')}
        Document Abstract/Introduction: {document_text}
        
        Return ONLY a JSON array of domains found (empty array if no industry domains found):
        {{"domains": ["domain1", "domain2", "domain3"]}}
        
        DO NOT include ANY GENERIC TERMS LIKE "AI", "RESEARCH", "GENERAL", ETC. in the JSON array.
        """
        
        try:
            # Call DeepSeek LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            response_json = json.loads(response.choices[0].message.content)
            
            # Validate response structure
            if 'domains' not in response_json:
                raise ValueError("Response missing 'domains' key")
            
            if not isinstance(response_json['domains'], list):
                raise ValueError("'domains' value must be a list")
            
            return response_json
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}")
        except Exception as e:
            raise ValueError(f"Domain extraction failed: {e}")
        
