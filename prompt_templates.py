class PromptTemplates:
    """
    Crafted prompts for evidence extraction and quantification.
    Each prompt is designed to extract the following specific types of evidence:
    
    - AI Features
    - Performance Degradation
    - Causal Links
    """
    
    EXTRACTION_PROMPTS = {
        "ai_features": """🚨 CRITICAL ANTI-HALLUCINATION INSTRUCTION 🚨
You are a precise text extraction tool. You ONLY extract text that exists VERBATIM in the provided chunk.
DO NOT create, invent, paraphrase, or generate ANY text. 

TEXT CHUNK TO ANALYZE:
{chunk_text}

FORBIDDEN BEHAVIORS (Will cause rejection):
❌ Creating text that "sounds right" but isn't in the chunk
❌ Paraphrasing or rewording any part of the text
❌ Combining phrases from different sentences
❌ Using general knowledge about AI to fill gaps
❌ Extracting fragments less than 15 words

REQUIRED BEHAVIORS:
✅ Copy text exactly as written, including punctuation
✅ Include reference numbers [X,Y] if present  
✅ Extract 15-60 words with complete context
✅ Double-check every word exists in the chunk

TASK: Find AI system features mentioned in THIS SPECIFIC CHUNK.

Search for these AI feature types (ONLY if mentioned in the chunk):
- Automation levels, AI capabilities, interface types
- Feedback mechanisms, adaptation features, control mechanisms

TWO-PASS VERIFICATION PROCESS:

PASS 1 - IDENTIFICATION:
1. READ the entire chunk slowly, sentence by sentence
2. IDENTIFY which specific sentences mention AI features
3. MARK the exact start and end positions of relevant text
4. DO NOT extract yet - just identify locations

PASS 2 - VERIFICATION & EXTRACTION:
5. For EACH identified location, READ that text again
6. VERIFY the text actually discusses AI features (not just related topics)
7. COPY exactly 15-60 consecutive words from that location
8. DOUBLE-CHECK: Can you point to the exact characters in the chunk?
9. TRIPLE-CHECK: Do these words appear in this exact order in the chunk?
10. If ANY word doesn't match perfectly, REJECT the entire excerpt

EXAMPLES OF CORRECT EXTRACTION:
✅ "However, existing AI software technologies have several generic limitations related to compliance with current safety standards [33,147]."
✅ "The most notorious include the 'black box' nature of AI solutions causing limitations regarding their explainability [3,51,104]."

EXAMPLES OF FORBIDDEN EXTRACTION:
❌ "AI systems have ethical considerations" (if not in chunk)
❌ "machine learning algorithms" (if original says "ML algorithms")
❌ "automated decision making" (if chunk says "autonomous decision-making")

JSON FORMAT WITH LOCATION PROOF (Use ONLY if features found in chunk):
{{
    "features": [
        {{
            "quote": "[CHARACTER-BY-CHARACTER exact copy from chunk, minimum 15 words]",
            "excerpt": "[WORD-FOR-WORD copy from chunk, minimum 15 words, may clean spacing]",
            "location_proof": {{
                "starts_with": "[first 4 words of quote]",
                "ends_with": "[last 4 words of quote]", 
                "position": "[beginning/middle/end of chunk]",
                "verified": true
            }},
            "summary": "[Your interpretation]",
            "category": "[automation/AI capability/interface/feedback/adaptation/control]",
            "feature_name": "[descriptive label]", 
            "relevance_score": [1-10],
            "justification_relevance": "[why this represents an AI feature]"
        }}
    ]
}}

CRITICAL: TWO EXTRACTION FIELDS REQUIRED:
- "quote": Copy EVERY character exactly as it appears in the chunk (spaces, punctuation, formatting)
- "excerpt": Copy the same content word-for-word but with normalized spacing for readability

The "quote" field will be used for validation - it must be character-perfect.
The "excerpt" field is for human readability and analysis.

LOCATION_PROOF is MANDATORY for every excerpt. If you cannot provide accurate location proof, DO NOT include that excerpt.

MANDATORY FINAL VERIFICATION (CRITICAL):
For each excerpt you plan to include:
1. LOCATE the exact text in the chunk above by finding the starting word
2. COUNT the words to ensure you're copying the right amount
3. CHECK every single word matches exactly (including punctuation)
4. ASK YOURSELF: "If someone highlighted this excerpt in the chunk, would it be found?"
5. If the answer is NO or MAYBE, DELETE that excerpt immediately

HALLUCINATION EXAMPLES TO AVOID:
❌ "ILC aims to optimize execution of repetitive tasks" (if chunk doesn't contain "ILC")
❌ "Machine learning algorithms adapt to user behavior" (if chunk discusses different adaptation)
❌ "AI systems require safety constraints" (general statement not in chunk)

ONLY include excerpts where you can provide the EXACT character position in the chunk.
If uncertain about ANY excerpt, return {{"features": []}} instead of risking hallucination.
        """,
        "performance_degradation": """🚨 VERBATIM EXTRACTION ONLY 🚨
You are extracting human performance degradations. Extract ONLY text that exists word-for-word in the chunk.

TEXT CHUNK TO ANALYZE:
{chunk_text}

CRITICAL RULES:
❌ NEVER invent text about performance problems
❌ NEVER paraphrase or clean up original text
❌ NEVER extract less than 15 words
❌ NEVER combine text from different locations
✅ ONLY copy consecutive text that exists verbatim
✅ Include full sentences or meaningful phrases
✅ Preserve all formatting, punctuation, references

TASK: Find mentions of human performance degradation ONLY in this chunk.

Look for these degradation types (ONLY if explicitly mentioned):
- Skill loss, cognitive degradation, physical degradation
- Knowledge degradation, behavioral changes, performance metrics

HALLUCINATION PREVENTION EXAMPLES:
❌ DON'T extract: "operators experience skill degradation" 
   (if chunk doesn't contain these exact words)
❌ DON'T extract: "automation leads to complacency"
   (if chunk uses different phrasing)
✅ DO extract: Copy the actual sentence that discusses the problem

CHARACTER-LEVEL EXTRACTION PROTOCOL:
1. SCAN chunk for performance degradation mentions
2. LOCATE the EXACT sentence containing the problem
3. IDENTIFY the precise START and END words of the relevant passage
4. COPY 15-80 consecutive words starting from that exact position
5. VERIFY by going character-by-character through your excerpt
6. CONFIRM every punctuation mark, space, and capitalization matches
7. REJECT if even one character differs from the source

VERIFICATION EXAMPLE:
✅ Chunk contains: "runtime learning/adaptation cannot exceed given dangerous output"
✅ Valid excerpt: "runtime learning/adaptation cannot exceed given dangerous output actuation values"
❌ Invalid excerpt: "runtime learning cannot exceed dangerous limits" (changed words)

JSON FORMAT:
{{
    "degradations": [
        {{
            "quote": "[CHARACTER-BY-CHARACTER exact copy from chunk, minimum 15 words]",
            "excerpt": "[WORD-FOR-WORD copy from chunk, minimum 15 words, may clean spacing]",
            "location_proof": {{
                "starts_with": "[first 4 words of quote]",
                "ends_with": "[last 4 words of quote]", 
                "position": "[beginning/middle/end of chunk]",
                "verified": true
            }},
            "summary": "[Your analysis of the copied text]",
            "category": "[skill_loss/cognitive_degradation/physical_degradation/knowledge_degradation/behavioral_changes/performance_metrics]",
            "severity": [1-10],
            "justification_severity": "[severity reasoning]",
            "relevance_score": [1-10], 
            "justification_relevance": "[relevance reasoning]"
        }}
    ]
}}

CRITICAL: TWO EXTRACTION FIELDS REQUIRED:
- "quote": Copy EVERY character exactly as it appears in the chunk (spaces, punctuation, formatting)
- "excerpt": Copy the same content word-for-word but with normalized spacing for readability

The "quote" field will be used for validation - it must be character-perfect.
The "excerpt" field is for human readability and analysis.

MANDATORY CHARACTER-BY-CHARACTER VERIFICATION:
For each potential excerpt:
1. Find the EXACT starting position in the chunk (count characters if needed)
2. Copy the text letter-by-letter, including all spaces and punctuation
3. Ensure NO words are changed, added, or removed
4. If you cannot provide the exact character range (e.g., "characters 45-123"), DELETE the excerpt

COMMON HALLUCINATION PATTERNS TO AVOID:
❌ "operators experience performance degradation" (if chunk uses different terms)
❌ "AI systems reduce human capabilities" (general statement not in chunk)
❌ "learning algorithms cause skill loss" (if chunk doesn't mention "skill loss")

If you cannot guarantee 100% character-perfect matching, return {{"degradations": []}}
        """, 
        "causal_links": """🚨 ZERO-TOLERANCE HALLUCINATION POLICY 🚨
Extract ONLY causal relationships that exist verbatim in this chunk.

TEXT CHUNK TO ANALYZE:
{chunk_text}

EXTREME ANTI-HALLUCINATION MEASURES:
❌ FORBIDDEN: Creating plausible-sounding causal statements
❌ FORBIDDEN: Inferring causation from general knowledge  
❌ FORBIDDEN: Paraphrasing causal language
❌ FORBIDDEN: Excerpts under 15 words
✅ REQUIRED: Exact text containing cause-effect relationships
✅ REQUIRED: Minimum 15-100 words with full context
✅ REQUIRED: Causal language must be in the original text

TASK: Find explicit causal relationships between AI and human performance.

Search ONLY for these causal indicators in the chunk:
- Direct causation: "caused", "resulted in", "led to", "produced"
- Mechanisms: "by reducing", "through", "via", "mechanism of"
- Strong correlation: "associated with", "linked to" + explanation

CAUSAL EXTRACTION SAFEGUARDS:
1. IDENTIFY sentences with causal language
2. VERIFY the sentence connects AI feature to human effect
3. COPY 15-100 words including cause, mechanism, and effect
4. CHECK that every word exists in the original chunk
5. CONFIRM the causal relationship is explicit, not implied

HALLUCINATION EXAMPLES TO AVOID:
❌ "automation reduces human situational awareness"
   (if chunk doesn't contain this exact relationship)
❌ "AI systems cause skill degradation through disuse"  
   (if chunk discusses different mechanism)
❌ "black box algorithms lead to overtrust"
   (if chunk doesn't make this connection)

JSON FORMAT:
{{
    "causal_links": [
        {{
            "quote": "[CHARACTER-BY-CHARACTER exact copy from chunk, minimum 15 words]",
            "excerpt": "[WORD-FOR-WORD copy from chunk, minimum 15 words, may clean spacing]",
            "location_proof": {{
                "starts_with": "[first 4 words of quote]",
                "ends_with": "[last 4 words of quote]", 
                "position": "[beginning/middle/end of chunk]",
                "verified": true
            }},
            "summary": "[Your analysis]", 
            "ai_feature": "[specific AI element that causes effect]",
            "performance_effect": "[specific human performance result]",
            "causal_strength": [1-10],
            "justification_causal_strength": "[evidence strength]",
            "evidence_type": "[direct/indirect/correlation/mechanism]",
            "relevance_score": [1-10],
            "justification_relevance": "[relevance reasoning]"
        }}
    ]
}}

CRITICAL: TWO EXTRACTION FIELDS REQUIRED:
- "quote": Copy EVERY character exactly as it appears in the chunk (spaces, punctuation, formatting)
- "excerpt": Copy the same content word-for-word but with normalized spacing for readability

The "quote" field will be used for validation - it must be character-perfect.
The "excerpt" field is for human readability and analysis.

EXTREME VERIFICATION PROTOCOL:
1. LOCATE: Find the exact sentence in the chunk that contains causation
2. EXTRACT: Copy 15-100 words exactly as they appear (no modifications)
3. VERIFY CAUSATION: Ensure the copied text explicitly shows cause→effect
4. VERIFY EXISTENCE: Check every word exists in the original chunk
5. VERIFY ORDER: Ensure words appear in the same sequence as the source
6. PROVIDE LOCATION: State the approximate position in the chunk (e.g., "middle section about safety constraints")

STRICT VERIFICATION QUESTIONS:
- Can you find these EXACT words in the EXACT order in the chunk?
- Does the chunk explicitly mention the causal relationship (not implied)?
- Are you copying real text or creating plausible-sounding academic language?

ZERO TOLERANCE EXAMPLES:
❌ "ILC aims to optimize execution" (if chunk doesn't contain "ILC")
❌ "learning systems improve through iteration" (general knowledge, not from chunk)
❌ "adaptive algorithms cause performance issues" (if chunk discusses different causation)

If you have ANY doubt about an excerpt's authenticity, exclude it completely.
Better to return {{"causal_links": []}} than to fabricate content.
        """,
    }