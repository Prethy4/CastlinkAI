import chromadb
from chromadb.utils import embedding_functions
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import List
from database import SessionLocal, Talent
from schemas import AgentState
from config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_CHAT_MODEL,
    SYSTEM_PROMPT,
)

# --- CHROMADB SETUP ---
chroma_client = chromadb.Client()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=OPENAI_EMBEDDING_MODEL,
)
collection = chroma_client.get_or_create_collection(name="talent_pool", embedding_function=openai_ef)

@tool
def search_talent(query: str = "", location: str = None, continent: str = None, country: str = None, job_type: str = None, 
                  gender: str = None, hair_color: str = None, eye_color: str = None, skin_color: str = None,
                  budget: int = None, shoot_dates: List[str] = None):
    """
    Searches for talent in the database.
    Use 'query' for general semantic search.
    Use specific arguments for structured criteria: 'location', 'continent', 'country', 'job_type', 'gender', 'hair_color', 'eye_color', 'skin_color', 'budget', 'shoot_dates'.
    Always accumulate criteria from the conversation history.
    """
    db = SessionLocal()
    query_obj = db.query(Talent)

    # Apply all available structured filters first
    if location:
        query_obj = query_obj.filter(Talent.location.ilike(f"%{location}%"))
    if continent:
        query_obj = query_obj.filter(Talent.continent.ilike(f"%{continent}%"))
    if country:
        query_obj = query_obj.filter(Talent.country.ilike(f"%{country}%"))
    if gender:
        query_obj = query_obj.filter(Talent.gender.ilike(f"%{gender}%"))
    if hair_color:
        query_obj = query_obj.filter(Talent.hair.ilike(f"%{hair_color}%"))
    if eye_color:
        query_obj = query_obj.filter(Talent.eyes.ilike(f"%{eye_color}%"))
    if skin_color:
        query_obj = query_obj.filter(Talent.skin_color.ilike(f"%{skin_color}%"))
    if budget is not None:
        query_obj = query_obj.filter(Talent.budget_tier <= budget)

    # If a semantic query is provided, used for further refine of results
    if query:
        chroma_where = {}
        if location:
       
            chroma_where["location"] = location

        results = collection.query(
            query_texts=[query],
            n_results=50,
            where=chroma_where if chroma_where else None
        )
        
        if results and results['ids'] and results['ids'][0]:
            talent_ids = results['ids'][0]
            query_obj = query_obj.filter(Talent.id.in_(talent_ids))
        else:
            db.close()
            return []

    talents = query_obj.all()
    
    result_list = []
    for t in talents:
        if job_type and t.job_types:
            j_types = t.job_types if isinstance(t.job_types, list) else str(t.job_types)
            if job_type not in j_types:
                continue
        
        if shoot_dates and t.availability:
            talent_availability = t.availability if isinstance(t.availability, list) else [str(t.availability)] 
            valid_shoot_dates = [date for date in shoot_dates if date]
            if valid_shoot_dates and not any(date in talent_availability for date in valid_shoot_dates):
                continue
            
        photos = t.photos[0] if t.photos and isinstance(t.photos, list) and len(t.photos) > 0 else None

        result_list.append({
            "id": t.id,
            "name": t.name,
            "photos": photos,
            "height": t.height,
            "bust": t.bust,
            "waist": t.waist,
            "hips": t.hips,
            "dress_size": t.dress_size,
            "hair": t.hair,
            "eyes": t.eyes,
            "agent_name": t.agent_name,
            "budget_tier": t.budget_tier,
        })
        if len(result_list) >= 10:
            break
    db.close()
    return result_list

def reasoner_node(state: AgentState):
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    tools = [search_talent]
    llm_with_tools = llm.bind_tools(tools)
    
    prompt_content = SYSTEM_PROMPT.format(filters=state.filters)
    sys_msg = SystemMessage(content=prompt_content)
    
    messages = [sys_msg] + state.messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode([search_talent])

# Build Graph
graph = StateGraph(AgentState)
graph.add_node("agent", reasoner_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    lambda x: "tools" if x.messages[-1].tool_calls else END,
    {"tools": "tools", END: END}
)
graph.add_edge("tools", "agent") 

app_graph = graph.compile()
