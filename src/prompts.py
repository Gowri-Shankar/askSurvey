"""Prompt templates for classification and RAG."""

TOPIC_LABELS = [
    "Product Durability",
    "Material/Build quality",
    "Functionality & Ease of use",
    "Product Performance (Works well)",
    "Authenticity (e.g., original vs counterfeit)",
    "Style / design / Size / fit / look satisfaction",
    "Matches images",
    "Matches specifications/details",
    "Value for money",
    "Overpriced vs. budget-friendly",
    "Discount/Sale satisfaction",
    "Hidden Fee Charged",
    "New Type of Fee Charged",
    "High Fee Charged",
    "Shipping speed",
    "Presentation & Packaging condition",
    "Delivery accuracy",
    "Shipping cost",
    "Courier service experience",
    "Seller responsiveness",
    "Seller helpfulness & resolution of issues",
    "Platform customer service responsiveness",
    "Platform customer service helpfulness & resolution of issues",
    "Availability issues & back-order experience",
    "Clarity of Return policy",
    "Ease of return/exchange process",
    "Refund processing time",
    "Platform first impressions",
    "Platform ease of navigation",
    "Platform buying experience",
    "Platform paying experience",
    "Platform promotions communication",
    "Platform communication means",
    "Platform - Generic",
    "Product - Generic",
]

SENTIMENT_LABELS = ["Positive", "Neutral", "Negative"]


def build_classification_prompt(review_text: str) -> str:
    """Build the classification prompt for a review."""
    topics_str = ", ".join(TOPIC_LABELS)

    prompt = f"""### Instructions:
Based on the survey text response: '{review_text}', choose exactly one Topic from the following list that best represents the dominant theme of the review: {topics_str}.
The dominant topic is the one most prominently discussed or the main point of the review.
Then, determine the overall sentiment of the review: Positive, Neutral, or Negative. If the review has mixed sentiments, consider the sentiment related to the dominant topic; if significant negative aspects are present, prioritize Negative sentiment.

Please only provide one classification label in the format: **Topic--Sentiment** and **nothing else**. **Do not explain, do not provide reasoning**"""

    return prompt


RAG_PROMPT_TEMPLATE = """As a highly knowledgeable product review assistant, your role is to accurately interpret product-related queries and provide responses using our specialized product review database. Follow these directives to ensure optimal user interactions:

1. Precision in Answers: Respond solely with information directly relevant to the user's query from our product review database. Avoid assumptions or adding unrelated details.
2. Topic Relevance: Limit your expertise to specific product-related areas:
   - Product Features and Specifications
   - Customer Feedback and Ratings
   - Pros and Cons Highlighted in Reviews
   - Usage Tips and Common Issues
3. Handling Off-topic Queries: For questions unrelated to products or reviews (e.g., "What is the capital of France?"), politely inform the user that the query is outside the chatbot's scope and suggest focusing on product-related inquiries.
4. Promoting Informed Decisions: Craft responses that help users make well-informed choices based on authentic customer experiences.
5. Contextual Accuracy: Ensure responses are directly related to the product query, utilizing only pertinent information from our review database.
6. Relevance Check: If a query does not align with our product review database, guide the user to refine their question or politely decline to provide an answer.
7. Avoiding Duplication: Ensure no response is repeated within the same interaction, maintaining uniqueness and relevance to each user query.
8. Streamlined Communication: Eliminate any unnecessary comments or closing remarks from responses. Focus on delivering clear, concise, and direct answers.
9. Avoid Non-essential Sign-offs: Do not include any sign-offs like "Best regards" or "ReviewBot" in responses.
10. One-time Use Phrases: Avoid using the same phrases multiple times within the same response. Each sentence should be unique and contribute to the overall message without redundancy.
11. If you don't find relevant answers, don't make up facts, stick to the things that are present in the database.

Also, provide a concise output, not in bits and pieces, make sure that you follow a proper alignment.

Product Query Context:
{context}

Question: {question}

Answer:"""


ROUTER_SYSTEM_PROMPT = """You are a routing assistant. You must choose exactly one label: `rerank_insight` or `pandas_metric`."""

ROUTER_EXAMPLES = [
    ("How many negative reviews are there?", "pandas_metric"),
    ("What percentage of positive reviews did we get?", "pandas_metric"),
    ("What are the most common complaints about delivery?", "rerank_insight"),
    ("Summarize the top pain points users mention.", "rerank_insight"),
]


def build_router_prompt(question: str) -> list:
    """Build the router prompt with examples."""
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
    except ImportError:
        from langchain.schema import SystemMessage, HumanMessage

    messages = [SystemMessage(content=ROUTER_SYSTEM_PROMPT)]

    for q, lbl in ROUTER_EXAMPLES:
        messages.append(HumanMessage(content=f"Q: {q}"))
        messages.append(HumanMessage(content=f"A: {lbl}"))

    messages.append(HumanMessage(content=f"Q: {question}"))
    messages.append(HumanMessage(content="A:"))

    return messages
