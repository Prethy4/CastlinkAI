from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import List, Dict, Any, Annotated
from pydantic import BaseModel
from sqlalchemy import or_, cast, String
from database import SessionLocal, Talent
from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL, SYSTEM_PROMPT

class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    filters: Dict[str, Any] = {}
    found_talents: List[Dict] = []

@tool
def generate_casting(location: str = None, continent: str = None, country: str = None, job_type: str = None,
                     gender: str = None, hair_color: str = None, eye_color: str = None, skin_color: str = None,
                     budget: int = None, shoot_dates: List[str] = None, category: str = None, hair_type: str = None,
                     height: str = None, bust: str = None, waist: str = None, hips: str = None, dress_size: str = None,
                     shoe_size: str = None):
    """
    Search for talent. Only call when Gender, Category, Location, Job Type, and Shoot Dates are known.
    """
    db = SessionLocal()
    query_obj = db.query(Talent)

    # Mandatory & Semi-Mandatory Filters (approximated match)
    if location:
        query_obj = query_obj.filter(
            or_(
                Talent.location.ilike(f"%{location}%"),
                Talent.country.ilike(f"%{location}%"),
                Talent.continent.ilike(f"%{location}%")
            )
        )
    if continent:
        query_obj = query_obj.filter(Talent.continent.ilike(f"%{continent}%"))
    if country:
        query_obj = query_obj.filter(Talent.country.ilike(f"%{country}%"))
    if gender:
        query_obj = query_obj.filter(Talent.gender.ilike(f"%{gender}%"))
    if category:
        query_obj = query_obj.filter(Talent.category.ilike(f"%{category}%"))
    if hair_color:
        query_obj = query_obj.filter(Talent.hair.ilike(f"%{hair_color}%"))
    if hair_type:
        query_obj = query_obj.filter(Talent.hair_type.ilike(f"%{hair_type}%"))
    if eye_color:
        query_obj = query_obj.filter(Talent.eyes.ilike(f"%{eye_color}%"))
    if skin_color:
        query_obj = query_obj.filter(Talent.skin_color.ilike(f"%{skin_color}%"))
    if budget is not None:
        query_obj = query_obj.filter(Talent.budget_tier <= budget)
    if job_type:
        # Optimization: Filter job_type in SQL instead of Python to reduce fetch size
        query_obj = query_obj.filter(cast(Talent.job_types, String).ilike(f"%{job_type}%"))

    # Non-Mandatory Filters (exact match)
    if height:
        query_obj = query_obj.filter(Talent.height == height)
    if bust:
        query_obj = query_obj.filter(Talent.bust == bust)
    if waist:
        query_obj = query_obj.filter(Talent.waist == waist)
    if hips:
        query_obj = query_obj.filter(Talent.hips == hips)
    if dress_size:
        query_obj = query_obj.filter(Talent.dress_size == dress_size)
    if shoe_size:
        query_obj = query_obj.filter(Talent.shoe_size == shoe_size)

    # Optimization: Limit fetch to 50 to prevent memory overflow with 1000 users
    talents = query_obj.limit(50).all()
    
    result_list = []
    for t in talents:
        if shoot_dates and t.availability:
            # Date logic is complex, keeping in Python but operating on a smaller dataset now
            talent_availability = t.availability if isinstance(t.availability, list) else [str(t.availability)] 
            valid_shoot_dates = [date for date in shoot_dates if date]
            if valid_shoot_dates and not any(date in talent_availability for date in valid_shoot_dates):
                continue
            
        photos = t.photos[0] if t.photos and isinstance(t.photos, list) and len(t.photos) > 0 else None

        result_list.append({
            "id": t.id,
            "name": t.name,
            "availability": t.availability,
            "photos": photos,
            "height": t.height,
            "bust": t.bust,
            "waist": t.waist,
            "hips": t.hips,
            "dress_size": t.dress_size,
            "shoe_size": t.shoe_size,
            "hair": t.hair,
            "hair_type": t.hair_type,
            "eyes": t.eyes,
            "agent_name": t.agent_name,
            "budget_tier": t.budget_tier,
        })
        if len(result_list) >= 10:
            break
    db.close()
    return result_list

def reasoner_node(state: AgentState):
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0, api_key=OPENAI_API_KEY, max_retries=3)
    tools = [generate_casting]
    llm_with_tools = llm.bind_tools(tools)
    
    prompt_content = SYSTEM_PROMPT.format(filters=state.filters)
    sys_msg = SystemMessage(content=prompt_content)
    
    messages = [sys_msg] + state.messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def custom_tool_node(state: AgentState):
    outputs = []
    for tool_call in state.messages[-1].tool_calls:
        if tool_call["name"] == "generate_casting":
            # Execute the tool
            result = generate_casting.invoke(tool_call["args"])
            
            # Create a concise summary for the LLM to save tokens
            # The full data is passed in 'artifact' which the LLM does not see, but the UI can read
            outputs.append(ToolMessage(
                content=f"Search completed. Found {len(result)} talents. The full list has been sent to the user interface.",
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                artifact=result
            ))
    return {"messages": outputs}

# Build Graph
graph = StateGraph(AgentState)
graph.add_node("agent", reasoner_node)
graph.add_node("tools", custom_tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    lambda x: "tools" if x.messages[-1].tool_calls else END,
    {"tools": "tools", END: END}
)
graph.add_edge("tools", "agent") 

app_graph = graph.compile()
