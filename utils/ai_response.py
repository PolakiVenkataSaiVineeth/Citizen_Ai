import random
import json
import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Mock responses for different categories (used as fallback when API is not available)
MOCK_RESPONSES = {
    "greeting": [
        "Hello! How can I assist you with government services today?",
        "Welcome to Citizen AI! I'm here to help with your civic needs.",
        "Greetings! I'm your AI assistant for government-related questions."
    ],
    "general_info": [
        "I can provide information on various government services including permits, licenses, taxes, and public facilities.",
        "As your civic assistant, I can help you navigate government processes and find the resources you need.",
        "I'm designed to make government services more accessible and easier to understand."
    ],
    "help": [
        "You can ask me about government services, how to apply for permits, where to find public facilities, or submit feedback about civic issues.",
        "I can assist with questions about local government, direct you to the right department, or help you understand civic processes.",
        "Feel free to ask about any government service or program you need help with."
    ],
    "fallback": [
        "I'm still learning about that topic. Could you provide more details so I can better assist you?",
        "I don't have specific information on that yet, but I'd be happy to help you find the right resources.",
        "That's a bit outside my current knowledge. Would you like me to direct you to a human representative who might know more?"
    ]
}

# Topic classification keywords
TOPIC_KEYWORDS = {
    "transportation": ["bus", "train", "subway", "transit", "transportation", "traffic", "road", "highway", "bike", "parking"],
    "healthcare": ["health", "hospital", "doctor", "medical", "clinic", "insurance", "vaccine", "medicine", "emergency"],
    "education": ["school", "education", "college", "university", "student", "teacher", "class", "course", "degree", "learning"],
    "housing": ["housing", "rent", "apartment", "house", "mortgage", "lease", "tenant", "landlord", "property"],
    "taxes": ["tax", "taxes", "refund", "filing", "deduction", "income", "property tax", "tax return"],
    "permits": ["permit", "license", "application", "approval", "certificate", "registration", "renewal"],
    "voting": ["vote", "voting", "election", "ballot", "register", "polling", "candidate", "campaign"],
    "public_safety": ["police", "fire", "emergency", "safety", "crime", "report", "security", "protection"]
}

# Domain-specific responses
DOMAIN_RESPONSES = {
    "transportation": [
        "Our city's transportation system includes buses, trains, and bike lanes. What specific information are you looking for?",
        "I can help you with transit schedules, fare information, or how to report transportation issues.",
        "The transportation department handles traffic management, public transit, and road maintenance. How can I assist you with transportation today?"
    ],
    "healthcare": [
        "Public health services include community clinics, vaccination programs, and health education resources.",
        "I can provide information about government health programs, finding local health facilities, or applying for health benefits.",
        "Our health department offers various services from preventive care to emergency response. What health information do you need?"
    ],
    "education": [
        "The education department oversees public schools, adult education programs, and educational resources.",
        "I can help you find information about school enrollment, educational programs, or learning resources provided by the government.",
        "Public education services include K-12 schools, libraries, and continuing education opportunities. What would you like to know about?"
    ],
    "housing": [
        "Government housing programs include affordable housing initiatives, rental assistance, and homebuyer support.",
        "I can provide information about public housing applications, tenant rights, or housing assistance programs.",
        "The housing department handles issues related to affordable housing, homelessness prevention, and property standards."
    ],
    "taxes": [
        "I can help you understand local tax obligations, find tax forms, or learn about available tax assistance programs.",
        "Tax services include property tax assessment, payment options, and tax relief programs for eligible citizens.",
        "The revenue department handles tax collection, assessments, and can provide guidance on tax-related questions."
    ],
    "permits": [
        "Many activities require government permits, including construction, events, and certain business operations.",
        "I can guide you through the permit application process, help you understand requirements, or direct you to the right department.",
        "Permit and licensing services ensure safety and compliance with local regulations. What type of permit are you interested in?"
    ],
    "voting": [
        "Voter services include registration, finding polling locations, and information about upcoming elections.",
        "I can help you check your voter registration status, find your polling place, or understand the voting process.",
        "The election department ensures fair and accessible voting. How can I assist you with voting-related questions?"
    ],
    "public_safety": [
        "Public safety services include police, fire, emergency management, and community safety programs.",
        "I can help you report non-emergency issues, find safety resources, or connect with local safety officials.",
        "Our public safety departments work to keep the community safe through prevention, response, and education programs."
    ]
}

# Context tracking
context_store = {}

async def get_ai_response(message: str, session_id: Optional[str] = None, context: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Generate an AI response to the user's message using Gemini Flash 1.5 model.
    Falls back to mock responses if the API is not available.
    
    Args:
        message: The user's message
        session_id: Unique identifier for the conversation session
        context: Previous messages in the conversation
        
    Returns:
        AI-generated response
    """
    # Initialize or retrieve context for this session
    if not session_id:
        session_id = "default"
    
    if session_id not in context_store:
        context_store[session_id] = []
    
    # Add current message to context
    if context:
        context_store[session_id] = context
    context_store[session_id].append({"role": "user", "content": message})
    
    # Try to use Gemini API if available
    if GEMINI_API_KEY:
        try:
            # Format conversation history for Gemini
            formatted_history = []
            for msg in context_store[session_id]:
                role = "user" if msg["role"] == "user" else "model"
                formatted_history.append({"role": role, "parts": [msg["content"]]})
            
            # Initialize Gemini model
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Generate response
            chat = model.start_chat(history=formatted_history[:-1] if len(formatted_history) > 1 else [])
            response = chat.send_message(message)
            ai_response = response.text
            
            # Add response to context
            context_store[session_id].append({"role": "assistant", "content": ai_response})
            return ai_response
            
        except Exception as e:
            print(f"Error using Gemini API: {str(e)}")
            # Fall back to mock responses
            pass
    
    # If API call failed or API key not available, use mock responses
    message_lower = message.lower()
    
    # Check for greetings
    if any(word in message_lower for word in ["hello", "hi", "hey", "greetings"]):
        response = random.choice(MOCK_RESPONSES["greeting"])
    
    # Check for help requests
    elif any(word in message_lower for word in ["help", "assist", "support", "guide"]):
        response = random.choice(MOCK_RESPONSES["help"])
    
    # Check for domain-specific queries
    else:
        # Identify the domain based on keywords
        identified_domain = None
        for domain, keywords in TOPIC_KEYWORDS.items():
            if any(keyword in message_lower for keyword in keywords):
                identified_domain = domain
                break
        
        if identified_domain:
            response = random.choice(DOMAIN_RESPONSES[identified_domain])
        else:
            # Use general or fallback response
            response = random.choice(MOCK_RESPONSES["general_info"] if random.random() > 0.3 else MOCK_RESPONSES["fallback"])
    
    # Add response to context
    context_store[session_id].append({"role": "assistant", "content": response})
    
    return response