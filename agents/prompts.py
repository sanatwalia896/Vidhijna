legal_query_rewriter_instructions = """Your goal is to generate a concise, information-rich legal search query.
The query should be suitable for both web search engines (Google, Bing) and Indian legal research databases (Manupatra, Indian Kanoon, SCC Online).

<TOPIC>
{research_topic}
</TOPIC>

<GUIDELINES>
- Use natural language in the form of a complete sentence or query.
- Include relevant legal terms, statutes, doctrines, or sections (e.g., IPC 498A, Article 21, Dowry Prohibition Act, 1961).
- Specify jurisdiction or temporal context if needed (e.g., Supreme Court of India, post-2013, recent rulings).
- Focus on a key legal issue, dispute, or remedy such as breach of trust, compensation, digital evidence, etc.
- Avoid keyword stuffing, repetition, or listing multiple years.
- Limit the query to 50 words or 400 characters.
</GUIDELINES>

<FORMAT>
Format your response as a JSON object with ALL three of these exact keys:
   - "query": The final legal search query string
   - "aspect": The specific legal angle or focus
   - "rationale": Brief explanation of why this formulation is effective
</FORMAT>

<EXAMPLE>
Example output:
{{
    "query": "Legal remedies under IPC 498A and Dowry Prohibition Act, 1961 for dowry-related abuse in India post-2013, with reference to Supreme Court judgments and digital evidence in recent case law",
    "aspect": "dowry-related criminal law and enforcement",
    "rationale": "Combines relevant statutes, time scope, and enforcement mechanisms for accurate legal research"
}}
</EXAMPLE>

Provide your response in JSON format:
"""


legal_summarizer_instructions = """
<GOAL>
Generate a clear, professional legal summary tailored to Indian legal research.
The summary should identify and organize the key legal dimensions of a topic or set of legal materials.
</GOAL>

<REQUIREMENTS>
When creating a NEW summary:
1. Identify and summarize the key legal issues and research questions.
2. Include applicable statutes, sections, and relevant case law (e.g., IPC, CrPC, Constitution of India).
3. Highlight legal doctrines, principles, and their applicability.
4. Outline potential arguments and counterarguments.
5. Mention jurisdictional considerations and procedural aspects.
6. Distinguish between binding and persuasive legal authorities.

When EXTENDING an existing summary:
1. Read the existing summary and any new legal material carefully.
2. Integrate new points as follows:
   a. If related to an existing issue, merge into the appropriate section.
   b. If it's a distinct legal angle, add a new logically placed paragraph.
   c. If irrelevant, exclude it.
3. Maintain coherence and legal accuracy.
4. Ensure the final output is not identical to the previous summary.
</REQUIREMENTS>

<FORMATTING>
- Write in a professional, objective tone suitable for legal analysis.
- Use clear subheadings or bullet points where appropriate.
- Do not use XML tags in the output; start directly with the summary.
</FORMATTING>
"""

legal_reflection_instructions = """You are an Indian legal research assistant reviewing a summary on: {research_topic}.

<GOAL>
1. Identify missing legal authorities, overlooked issues, or unclear doctrinal areas.
2. Generate a follow-up question to deepen legal analysis.
3. Focus on statutory gaps, case law ambiguities, procedural blind spots, or jurisdictional issues.
</GOAL>

<REQUIREMENTS>
- Ensure the follow-up query is self-contained and understandable without additional context.
- Use precise legal terminology relevant to Indian law (e.g., IPC, CrPC, constitutional articles, SC rulings).
- Limit the question to one key angle or unresolved issue.
- Make it suitable for web searches or legal databases like SCC Online or Indian Kanoon.
</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with these exact keys:
- knowledge_gap: Describe the missing or unclear legal dimension
- follow_up_query: Write a clear, targeted legal research question
</FORMAT>

<EXAMPLE>
Example output:
{{
    "knowledge_gap": "The summary does not address conflicting High Court interpretations on digital privacy under Article 21",
    "follow_up_query": "What are the key High Court rulings on digital privacy under Article 21 of the Indian Constitution?"
}}
</EXAMPLE>

Provide your output in JSON format:
"""
