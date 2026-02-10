class PromptTemplates:
    """
    Crafted prompts for evidence extraction and quantification.
    Each prompt is designed to extract the following specific types of evidence:
    
    - AI Features: Examine the mechanisms by which humans interact with autonomous systems, including both workflow-level interaction and cognitive-flow processes.
    - Performance Degradation: Examine the mechanisms by which humans interact with autonomous systems, including both workflow-level interaction and cognitive-flow processes.
    - Causal Links: Identify the types of human performance degradation that may arise from these new AI features, in contrast with traditional systems.
    - Measurables: Identify the measurable metrics for evaluating AI features that may degrade human performance during work, tasks, or operational activities, and explain how to use them. Examples include trust calibration indices, situation awareness scales (SAGAT, SART), cognitive workload measures (NASA-TLX, EEG-based metrics), error rate and error recovery time, latency in decision making, dependence or automation reliance indices, and transparency and explainability scores.
    - AI-interaction platforms: Review current research to determine whether real, semi-real, or simulated human-AI interaction platforms or frameworks exist. 
    """
    
    # Concise research context (reusable across prompts)
    RESEARCH_CONTEXT = """RESEARCH FOCUS (Human-AI Interaction):
Examining novel AI/autonomous system characteristics and features vs conventional automation in high-risk industries (healthcare, power, transportation, oil/gas, maritime, manufacturing):
(1) Non-deterministic/data-driven decision-making (outputs vary with data/model states → uncertainty + unpredictable failures)
(2) Opacity/lack of explainability (black-box behavior → trust/regulation challenges)
(3) Context-aware/adaptive behavior (dynamic environment response → variability/unpredictability)
"""
    
    EXTRACTION_PROMPTS = {
        "ai_features": """🚨 CRITICAL ANTI-HALLUCINATION INSTRUCTION 🚨
You are a precise text extraction tool. You ONLY extract text that exists VERBATIM in the provided chunk.
DO NOT create, invent, paraphrase, or generate ANY text. 

{RESEARCH_CONTEXT}

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

TASK: Find AI system features mentioned in THIS SPECIFIC CHUNK, prioritizing novel AI characteristics vs conventional automation.

PRIORITY SEARCH (ONLY if mentioned in chunk):
1. Non-deterministic/data-driven: variable outputs, model-dependent decisions, uncertainty, unpredictable failures
2. Opacity/explainability: black-box, lack of transparency, internal reasoning not visible, interpretability issues
3. Context-aware/adaptive: dynamic response, environment adaptation, real-time learning, variability handling
4. Platforms/frameworks: experimental testbeds, software tools, modeling approaches for human-AI interaction
5. General: automation levels, AI capabilities, interface types, feedback mechanisms, control mechanisms

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
            "prefix": "MUST FOLLOW THE EXACT FORMAT: AI Feature : [feature_name] : [category]"
            "location_proof": {{
                "starts_with": "[first 4 words of quote]",
                "ends_with": "[last 4 words of quote]", 
                "position": "[beginning/middle/end of chunk]",
                "verified": true
            }},
            "summary": "[Your interpretation of the excerpt]",
            "category": "[non_deterministic/opacity/context_adaptive/automation/AI_capability/interface/feedback/adaptation/control]",
            "feature_name": "[descriptive label of the AI feature that you have identified]", 
            "relevance_score": [1-10],
            "justification_relevance": "[why this excerpt mentions an AI feature]",
            "explanation": " MUST start with the prefix '[feature_name] : [category]' (use the actual values from this excerpt), then a colon and space, then the explanation. Format: '[feature_name] : [category] : [explanation]'. The explanation must: (1) Define the category and feature_name and clarify what each means. (2) Where possible, quote or reference the original wording used in the source (use exact phrases from the excerpt in quotes). Base the explanation ONLY on information present in the extracted excerpt - DO NOT introduce external knowledge or paraphrases not in the original text.",
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

{RESEARCH_CONTEXT}

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

TASK: Find mentions of human performance degradation ONLY in this chunk, focusing on novel AI-related degradations vs traditional automation.

PRIORITY SEARCH (ONLY if explicitly mentioned):
1. Trust issues: over-trust, under-trust, trust calibration, trust mismatch
2. Cognitive biases: automation bias, confirmation bias, bias in decision-making
3. Awareness/task issues: reduced situational awareness, mis-prioritization, mode confusion
4. Cognitive load: cognitive overload, interface complexity, information overload
5. Skill/knowledge: skill degradation, knowledge loss, skill atrophy, reduced competence
6. Interpretation: misinterpretation of AI recommendations, misunderstanding AI outputs
7. Evaluation metrics: trust calibration indices, situation awareness scales (SAGAT, SART), cognitive workload (NASA-TLX, EEG), error rates, decision latency, automation reliance, transparency scores
8. General: performance metrics, error rates, behavioral changes, workload measures

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
            "prefix": "MUST FOLLOW THE EXACT FORMAT: Performance Degradation : [category]",
            "location_proof": {{
                "starts_with": "[first 4 words of quote]",
                "ends_with": "[last 4 words of quote]", 
                "position": "[beginning/middle/end of chunk]",
                "verified": true
            }},
            "summary": "[Your analysis of the copied text]",
            "category": "[trust_issues/automation_bias/situational_awareness/cognitive_overload/skill_degradation/misinterpretation/performance_metrics/behavioral_changes]",
            "severity": [1-10],
            "justification_severity": "[severity reasoning]",
            "relevance_score": [1-10], 
            "justification_relevance": "[relevance reasoning]",
            "explanation": "MUST start with the prefix 'Degradation : [category]' (use the actual value from this excerpt), then a colon and space, then the explanation. Format: 'Degradation: [category] : [explanation]'. The explanation must: (1) Define the category and clarify what it means. (2) Where possible, quote or reference the original wording used in the source (use exact phrases from the excerpt in quotes). Base the explanation ONLY on information present in the extracted excerpt - DO NOT introduce external knowledge or paraphrases not in the original text.",
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

{RESEARCH_CONTEXT}

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

TASK: Find explicit causal relationships between AI features and human performance, especially how novel AI characteristics (non-deterministic, opacity, adaptive) causally affect human performance.

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
            "prefix": "MUST FOLLOW THE EXACT FORMAT: Causal Link : [ai_feature, degradation_type, evidence_type]",
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
            "ai_feature_type": "[non_deterministic/opacity/context_adaptive/other]",
            "degradation_type": "[trust_issues/automation_bias/situational_awareness/cognitive_overload/skill_degradation/misinterpretation/other]",
            "relevance_score": [1-10],
            "justification_relevance": "[relevance reasoning]",
            "explanation": "MUST start with the prefix 'Causal Link : [ai_feature, degradation_type]' (use the actual values from this excerpt), then a colon and space, then the explanation. Format: 'Causal Link: [ai_feature_type] : [explanation]'. The explanation must: (1) Define the ai_feature_type and clarify what it means. (2) Where possible, quote or reference the original wording used in the source (use exact phrases from the excerpt in quotes). Base the explanation ONLY on information present in the extracted excerpt - DO NOT introduce external knowledge or paraphrases not in the original text.",
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
        "measurables": """🚨 CRITICAL ANTI-HALLUCINATION INSTRUCTION 🚨
You are a precise text extraction tool. You ONLY extract text that exists VERBATIM in the provided chunk.
DO NOT create, invent, paraphrase, or generate ANY text.

{RESEARCH_CONTEXT}

TEXT CHUNK TO ANALYZE:
{chunk_text}

FORBIDDEN BEHAVIORS (Will cause rejection):
❌ Creating text about metrics that "sounds right" but isn't in the chunk
❌ Paraphrasing or rewording metric names or descriptions
❌ Combining phrases from different sentences
❌ Using general knowledge about metrics to fill gaps
❌ Extracting fragments less than 15 words

REQUIRED BEHAVIORS:
✅ Copy text exactly as written, including punctuation
✅ Include reference numbers [X,Y] if present  
✅ Extract 15-80 words with complete context
✅ Double-check every word exists in the chunk

TASK: Find measurable metrics mentioned in THIS SPECIFIC CHUNK for evaluating AI features that may degrade human performance during work, tasks, or operational activities, and explain how to use them.

PRIORITY SEARCH (ONLY if mentioned in chunk):
1. Trust metrics: trust calibration indices, trust scales, trust measurement methods, over-trust/under-trust measures
2. Situation awareness: SAGAT (Situation Awareness Global Assessment Technique), SART (Situation Awareness Rating Technique), SA measures
3. Cognitive workload: NASA-TLX (Task Load Index), EEG-based metrics, workload scales, cognitive load measures
4. Performance metrics: error rate, error recovery time, task completion time, accuracy measures
5. Decision metrics: latency in decision making, decision quality, response time, reaction time
6. Automation reliance: dependence indices, automation reliance measures, complacency metrics
7. Transparency metrics: explainability scores, interpretability measures, transparency ratings
8. Behavioral metrics: vigilance measures, monitoring behavior, engagement levels, interaction patterns
9. General: evaluation frameworks, assessment methods, measurement protocols, validation approaches

TWO-PASS VERIFICATION PROCESS:

PASS 1 - IDENTIFICATION:
1. READ the entire chunk slowly, sentence by sentence
2. IDENTIFY which specific sentences mention measurable metrics
3. MARK the exact start and end positions of relevant text
4. DO NOT extract yet - just identify locations

PASS 2 - VERIFICATION & EXTRACTION:
5. For EACH identified location, READ that text again
6. VERIFY the text actually discusses measurable metrics (not just general evaluation)
7. COPY exactly 15-80 consecutive words from that location
8. DOUBLE-CHECK: Can you point to the exact characters in the chunk?
9. TRIPLE-CHECK: Do these words appear in this exact order in the chunk?
10. If ANY word doesn't match perfectly, REJECT the entire excerpt

EXAMPLES OF CORRECT EXTRACTION:
✅ "Trust calibration was measured using the trust scale developed by Jian et al. (2000), which assesses appropriate reliance on automation."
✅ "Situation awareness was assessed using SAGAT at predetermined freeze points during the simulation tasks."
✅ "Cognitive workload was evaluated using the NASA-TLX questionnaire administered post-task."

EXAMPLES OF FORBIDDEN EXTRACTION:
❌ "Researchers measured trust levels" (if not in chunk)
❌ "NASA Task Load Index" (if original says "NASA-TLX")
❌ "error rates were calculated" (if chunk says "error rates were computed")

JSON FORMAT WITH LOCATION PROOF (Use ONLY if metrics found in chunk):
{{
    "measurables": [
        {{
            "quote": "[CHARACTER-BY-CHARACTER exact copy from chunk, minimum 15 words]",
            "excerpt": "[WORD-FOR-WORD copy from chunk, minimum 15 words, may clean spacing]",
            "prefix": "MUST FOLLOW THE EXACT FORMAT: Measurable : [metric_name] : [category] : [ai_feature] : [degradation_type]",
            "location_proof": {{
                "starts_with": "[first 4 words of quote]",
                "ends_with": "[last 4 words of quote]", 
                "position": "[beginning/middle/end of chunk]",
                "verified": true
            }},
            "summary": "[Your interpretation of the excerpt]",
            "category": "[trust_metrics/situation_awareness/cognitive_workload/performance_metrics/decision_metrics/automation_reliance/transparency_metrics/behavioral_metrics/other]",
            "ai_feature": "[specific AI feature that is being measured] if applicable, otherwise leave blank",
            "degradation_type": "[specific degradation type that is being measured, if applicable, otherwise leave blank [trust_issues/automation_bias/situational_awareness/cognitive_overload/skill_degradation/misinterpretation/other]",
            "metric_name": "[specific name of the metric, scale, or measure]",
            "measurement_method": "[how the metric is measured/applied, if described]",
            "relevance_score": [1-10],
            "justification_relevance": "[why this excerpt mentions a measurable metric]",
            "explanation": "MUST start with the prefix '[metric_name] : [category] : [ai_feature] : [degradation_type]' (use the actual values from this excerpt), then a colon and space, then the explanation. Format: '[metric_name] : [category] : [explanation]'. The explanation must: (1) Define the category and metric_name and clarify what each means. (2) Where possible, quote or reference the original wording used in the source (use exact phrases from the excerpt in quotes). (3) Explain how the metric is used or applied if that information is present in the excerpt. Base the explanation ONLY on information present in the extracted excerpt - DO NOT introduce external knowledge or paraphrases not in the original text.",
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
❌ "NASA-TLX was used to measure cognitive load" (if chunk doesn't mention NASA-TLX)
❌ "Trust was measured using Likert scales" (if chunk uses different measurement)
❌ "Error rates were calculated as performance metrics" (general statement not in chunk)

ONLY include excerpts where you can provide the EXACT character position in the chunk.
If uncertain about ANY excerpt, return {{"measurables": []}} instead of risking hallucination.
        """,
        "interaction_platforms": """🚨 CRITICAL ANTI-HALLUCINATION INSTRUCTION 🚨
You are a precise text extraction tool. You ONLY extract text that exists VERBATIM in the provided chunk.
DO NOT create, invent, paraphrase, or generate ANY text.

{RESEARCH_CONTEXT}

TEXT CHUNK TO ANALYZE:
{chunk_text}

FORBIDDEN BEHAVIORS (Will cause rejection):
❌ Creating text about platforms that "sounds right" but isn't in the chunk
❌ Paraphrasing or rewording platform names or descriptions
❌ Combining phrases from different sentences
❌ Using general knowledge about platforms to fill gaps
❌ Extracting fragments less than 15 words

REQUIRED BEHAVIORS:
✅ Copy text exactly as written, including punctuation
✅ Include reference numbers [X,Y] if present  
✅ Extract 15-100 words with complete context
✅ Double-check every word exists in the chunk

TASK: Find mentions of real, semi-real, or simulated human-AI interaction platforms or frameworks in THIS SPECIFIC CHUNK. These may include experimental testbeds, software tools, modeling approaches, or theoretical frameworks for studying human-autonomous system interaction.

PRIORITY SEARCH (ONLY if mentioned in chunk):
1. Platform/framework names: specific names of testbeds, simulators, software tools, frameworks
2. Appearance & structure: physical/virtual setup, architecture, components, interface design
3. Background: country, institution, organization, university, research group, development history
4. Functions & capabilities: what the platform can do, features, supported tasks, use cases
5. Operational workflow: input mechanisms, processing methods, output formats, interaction loops
6. Topics solved: problems addressed, research questions answered, validation results
7. Remaining challenges: limitations, open problems, technical difficulties, gaps
8. Future plans: planned developments, roadmap, upcoming features, research directions
9. Platform types: real testbed, semi-real simulation, virtual environment, theoretical model, hybrid approach

TWO-PASS VERIFICATION PROCESS:

PASS 1 - IDENTIFICATION:
1. READ the entire chunk slowly, sentence by sentence
2. IDENTIFY which specific sentences mention platforms/frameworks/testbeds
3. MARK the exact start and end positions of relevant text
4. DO NOT extract yet - just identify locations

PASS 2 - VERIFICATION & EXTRACTION:
5. For EACH identified location, READ that text again
6. VERIFY the text actually discusses interaction platforms (not just general AI systems)
7. COPY exactly 15-100 consecutive words from that location
8. DOUBLE-CHECK: Can you point to the exact characters in the chunk?
9. TRIPLE-CHECK: Do these words appear in this exact order in the chunk?
10. If ANY word doesn't match perfectly, REJECT the entire excerpt

EXAMPLES OF CORRECT EXTRACTION:
✅ "The SHERPA testbed was developed at MIT to study human-robot collaboration in manufacturing environments, featuring a modular architecture with real-time feedback mechanisms."
✅ "Researchers at TU Munich created the SafeAI simulator, which combines virtual reality with actual autonomous vehicle control systems to test operator interventions."
✅ "The platform supports real-time monitoring of pilot decisions during automated flight scenarios, recording gaze tracking, response times, and manual override actions."

EXAMPLES OF FORBIDDEN EXTRACTION:
❌ "The testbed was used for human-AI studies" (if not in chunk)
❌ "MIT developed a simulator" (if original says "researchers at MIT created a simulation environment")
❌ "The platform enables human-robot interaction testing" (general statement not in chunk)

JSON FORMAT WITH LOCATION PROOF (Use ONLY if platforms found in chunk):
{{
    "interaction_platforms": [
        {{
            "quote": "[CHARACTER-BY-CHARACTER exact copy from chunk, minimum 15 words]",
            "excerpt": "[WORD-FOR-WORD copy from chunk, minimum 15 words, may clean spacing]",
            "prefix": "MUST FOLLOW THE EXACT FORMAT: Interaction_platform : [platform_name] : [category]",
            "location_proof": {{
                "starts_with": "[first 4 words of quote]",
                "ends_with": "[last 4 words of quote]", 
                "position": "[beginning/middle/end of chunk]",
                "verified": true
            }},
            "summary": "[Your interpretation of the excerpt]",
            "category": "[real_testbed/semi_real_simulation/virtual_simulation/software_tool/theoretical_framework/modeling_approach/hybrid_platform/other]",
            "platform_name": "[specific name of the platform, framework, or testbed]",
            "platform_attributes": {{
                "background": "[country/institution/development history if mentioned]",
                "functions": "[capabilities and features if described]",
                "workflow": "[operational workflow if described]",
                "topics_solved": "[problems addressed if mentioned]",
                "challenges": "[remaining challenges if mentioned]",
                "future_plans": "[development plans if mentioned]"
            }},
            "relevance_score": [1-10],
            "justification_relevance": "[why this excerpt mentions an interaction platform]",
            "explanation": "MUST start with the prefix '[platform_name] : [category]' (use the actual values from this excerpt), then a colon and space, then the explanation. Format: '[platform_name] : [category] : [explanation]'. The explanation must: (1) Define the category and platform_name and clarify what each means. (2) Where possible, quote or reference the original wording used in the source (use exact phrases from the excerpt in quotes). (3) Describe any platform attributes that are mentioned in the excerpt (background, functions, workflow, etc.). Base the explanation ONLY on information present in the extracted excerpt - DO NOT introduce external knowledge or paraphrases not in the original text.",
        }}
    ]
}}

CRITICAL: TWO EXTRACTION FIELDS REQUIRED:
- "quote": Copy EVERY character exactly as it appears in the chunk (spaces, punctuation, formatting)
- "excerpt": Copy the same content word-for-word but with normalized spacing for readability

The "quote" field will be used for validation - it must be character-perfect.
The "excerpt" field is for human readability and analysis.

LOCATION_PROOF is MANDATORY for every excerpt. If you cannot provide accurate location proof, DO NOT include that excerpt.

PLATFORM_ATTRIBUTES INSTRUCTIONS:
- Only populate fields if that information is EXPLICITLY mentioned in the excerpt
- Leave fields empty if not mentioned - DO NOT infer or generate
- Each field should contain verbatim phrases from the excerpt, not summaries

MANDATORY FINAL VERIFICATION (CRITICAL):
For each excerpt you plan to include:
1. LOCATE the exact text in the chunk above by finding the starting word
2. COUNT the words to ensure you're copying the right amount
3. CHECK every single word matches exactly (including punctuation)
4. ASK YOURSELF: "If someone highlighted this excerpt in the chunk, would it be found?"
5. If the answer is NO or MAYBE, DELETE that excerpt immediately

HALLUCINATION EXAMPLES TO AVOID:
❌ "The CARLA simulator was used for autonomous driving research" (if chunk doesn't mention CARLA)
❌ "NASA developed the platform at Johnson Space Center" (if chunk uses different institution)
❌ "The testbed enables real-time human-robot interaction" (general statement not in chunk)

ONLY include excerpts where you can provide the EXACT character position in the chunk.
If uncertain about ANY excerpt, return {{"interaction_platforms": []}} instead of risking hallucination.
        """
    }