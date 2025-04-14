import streamlit as st
import nest_asyncio
from llama_index.core import Settings
from llama_index.llms.sambanovasystems import SambaNovaCloud
from llama_index.core.tools import FunctionTool
from llama_index.agent.lats import LATSAgentWorker
from llama_index.core.agent import AgentRunner
import http.client
import json
import os

# Apply nest_asyncio
nest_asyncio.apply()

# Product category configurations
PRODUCT_CATEGORIES = {
    "Cameras": {
        "features": [
            "Low Light Performance", "4K Video", "Image Stabilization",
            "Weather Sealing", "Compact Size", "WiFi Connectivity", "Touch Screen"
        ],
        "types": ["Mirrorless", "DSLR", "Point and Shoot", "Medium Format"],
        "use_cases": [
            "Professional Photography", "Vlogging", "Travel Photography",
            "Sports Photography", "Wildlife Photography"
        ]
    },
    "Laptops": {
        "features": [
            "Long Battery Life", "Dedicated Graphics", "Touch Screen",
            "Backlit Keyboard", "Fingerprint Reader", "Thunderbolt Ports", "5G Connectivity"
        ],
        "types": ["Ultrabook", "Gaming Laptop", "Business Laptop", "2-in-1 Convertible", "Budget Laptop"],
        "use_cases": ["Gaming", "Content Creation", "Business", "Student", "Programming"]
    },
    "Smartphones": {
        "features": [
            "5G Support", "Wireless Charging", "Water Resistance",
            "Face Recognition", "Multiple Cameras", "Fast Charging", "NFC"
        ],
        "types": ["Flagship", "Mid-range", "Budget", "Gaming Phone", "Compact"],
        "use_cases": ["Photography", "Gaming", "Business", "Basic Use", "Content Creation"]
    },
    "Smart Home Devices": {
        "features": [
            "Voice Control", "Mobile App Control", "Energy Monitoring",
            "Motion Detection", "Smart Scheduling", "Multi-user Support", "Integration Capabilities"
        ],
        "types": [
            "Smart Speakers", "Security Cameras", "Smart Lights",
            "Smart Thermostats", "Smart Displays"
        ],
        "use_cases": ["Home Security", "Energy Management", "Entertainment", "Home Automation", "Family Organization"]
    }
}

def initialize_llm():
    return SambaNovaCloud(
        model="Llama-4-Maverick-17B-128E-Instruct",
        context_window=10000,
        max_tokens=2048,
        temperature=0.1,
        top_k=1,
        top_p=0.95,
        additional_kwargs={"return_raw": True, "format_response": False}
    )

def search_with_serper(query: str, api_key: str) -> str:
    try:
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read()
        result = json.loads(data.decode("utf-8"))

        snippets = [item.get("snippet", "") for item in result.get("organic", [])[:4]]
        return "\n".join(snippets) if snippets else "No results found."

    except Exception as e:
        return f"Search failed: {str(e)}"

def setup_agent(serper_api_key: str):
    try:
        llm = initialize_llm()
        Settings.llm = llm

        search_tool = FunctionTool.from_defaults(
            fn=lambda q: search_with_serper(q, serper_api_key),
            name="search",
            description="Search for product information and reviews"
        )

        agent_worker = LATSAgentWorker(
            tools=[search_tool],
            num_expansions=2,
            max_rollouts=2,
            verbose=True,
            llm=llm
        )

        return AgentRunner(agent_worker)
    except Exception as e:
        st.error(f"Agent setup failed: {str(e)}")
        return None

def process_recommendation(query: str, agent: AgentRunner):
    try:
        chat_response = agent.chat(query)
        final_response = chat_response.response

        # Fallback: if LLM says “I am still thinking.” or response is too short
        if "I am still thinking." in final_response or len(final_response.strip()) < 50:
            try:
                return agent.list_tasks()[-1].extra_state["root_node"].children[0].children[0].current_reasoning[-1].observation
            except Exception:
                return final_response
        else:
            return final_response

    except Exception as e:
        return f"An error occurred while processing your request: {str(e)}"

def main():
    st.set_page_config(page_title="Smart Product Recommendation System", layout="wide")
    st.title("🎯 Smart Product Recommendation System")
    st.write("""
    Get personalized product recommendations based on your requirements. 
    Our AI-powered system analyzes current market offerings to find the best match for your needs.
    """)

    if 'agent' not in st.session_state:
        st.session_state.agent = None

    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter SambaNova API Key:", type="password")
        serper_api_key = st.text_input("Enter Serper API Key:", type="password")
        if api_key and serper_api_key:
            os.environ["SAMBANOVA_API_KEY"] = api_key
            if st.session_state.agent is None:
                st.session_state.agent = setup_agent(serper_api_key)

    st.header("What are you looking for?")
    category = st.selectbox("Select Product Category", list(PRODUCT_CATEGORIES.keys()))
    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("Budget (USD)", min_value=0, max_value=10000, value=1000)
    with col2:
        features = st.multiselect("Important Features", PRODUCT_CATEGORIES[category]["features"])

    custom_requirements = st.text_area("Any additional requirements or preferences?", height=100)

    if st.button("Get Recommendations", type="primary"):
        if not api_key or not serper_api_key:
            st.error("Please enter both your SambaNova and Serper API keys in the sidebar.")
            return
        if st.session_state.agent is None:
            st.error("Failed to initialize the recommendation agent. Please check your API keys and try again.")
            return

        try:
            query = f"Looking for a {category.lower()} under ${budget}"
            if features:
                query += f" with {', '.join(features)}"
            if custom_requirements:
                query += f". Additional requirements: {custom_requirements}"

            with st.spinner("Analyzing current market offerings..."):
                recommendation = process_recommendation(query, st.session_state.agent)

            st.header(f"🎯 Recommended {category}")
            st.write(recommendation)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    with st.expander("Need Help?"):
        st.write("""
        How to use this recommendation system:
        1. Enter your SambaNova and Serper API keys in the sidebar
        2. Select a product category
        3. Set your budget and preferences
        4. Specify your requirements
        5. Click 'Get Recommendations' to receive personalized suggestions
        """)

if __name__ == "__main__":
    main()
