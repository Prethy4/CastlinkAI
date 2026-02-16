from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import List, Dict, Any, Annotated, Optional
from pydantic import BaseModel, Field
from sqlalchemy import or_, cast, String
from database import SessionLocal, Talent
from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL, SYSTEM_PROMPT

class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    filters: Dict[str, Any] = {}
    found_talents: List[Dict] = []

class ExtractedFilters(BaseModel):
    location: Optional[str] = Field(None, description="Location")
    continent: Optional[str] = Field(None, description="Continent")
    country: Optional[str] = Field(None, description="Country")
    job_type: Optional[str] = Field(None, description="Job Type")
    gender: Optional[str] = Field(None, description="Gender")
    hair_color: Optional[str] = Field(None, description="Hair Color")
    eye_color: Optional[str] = Field(None, description="Eye Color")
    skin_color: Optional[str] = Field(None, description="Skin Color")
    budget: Optional[int] = Field(None, description="Budget")
    shoot_dates: Optional[List[str]] = Field(None, description="Shoot Dates")
    category: Optional[str] = Field(None, description="Category")
    hair_type: Optional[str] = Field(None, description="Hair Type")
    height: Optional[str] = Field(None, description="Height")
    bust: Optional[str] = Field(None, description="Bust")
    waist: Optional[str] = Field(None, description="Waist")
    hips: Optional[str] = Field(None, description="Hips")
    dress_size: Optional[str] = Field(None, description="Dress Size")
    shoe_size: Optional[str] = Field(None, description="Shoe Size")

def extract_information(user_input: str, current_filters: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts casting filters from user input using a lightweight LLM call."""
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    structured_llm = llm.with_structured_output(ExtractedFilters)
    
    prompt = f"""
    You are a Casting Assistant. Extract casting requirements from the user's message.
    Current known info: {current_filters}
    User message: "{user_input}"
    
    Return ONLY the fields that are explicitly mentioned or updated in the user message.
    If the user provides a role type like 'supporting', 'lead', or 'extra', map it to 'category'.
    """
    try:
        result = structured_llm.invoke(prompt)
        return {k: v for k, v in result.dict().items() if v is not None}
    except Exception as e:
        return {}

def generate_ask_response(missing_fields: List[str]) -> str:
    """Generates a polite question to ask for missing mandatory fields."""
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.6, api_key=OPENAI_API_KEY)
    prompt = f"""
    You are a Casting Assistant. You need to collect the following missing mandatory fields: {', '.join(missing_fields)}.
    Politely ask the user for ONE or TWO of these fields. Keep it short and professional.
    """
    return llm.invoke(prompt).content

@tool
def generate_casting(location: str = None, continent: str = None, country: str = None, job_type: str = None,
                     gender: str = None, hair_color: str = None, eye_color: str = None, skin_color: str = None,
                     budget: int = None, shoot_dates: List[str] = None, category: str = None, hair_type: str = None,
                     height: str = None, bust: str = None, waist: str = None, hips: str = None, dress_size: str = None,
                     shoe_size: str = None):
    """
    Search for talent. Returns ranked recommendations based on matched criteria.
    """
    db = SessionLocal()
    talents = db.query(Talent).all()
    
    scored_talents = []
    
    for t in talents:
        score = 0
        
        # Helper for case-insensitive partial match
        def matches(value, target):
            if not value or not target:
                return False
            return str(value).lower() in str(target).lower()

        if location:
            if matches(location, t.location) or matches(location, t.country) or matches(location, t.continent):
                score += 1
        if continent and matches(continent, t.continent): score += 1
        if country and matches(country, t.country): score += 1
        if gender and matches(gender, t.gender): score += 1
        if category and matches(category, t.category): score += 1
        if hair_color and matches(hair_color, t.hair): score += 1
        if hair_type and matches(hair_type, t.hair_type): score += 1
        if eye_color and matches(eye_color, t.eyes): score += 1
        if skin_color and matches(skin_color, t.skin_color): score += 1
        if height and t.height == height: score += 1
        if bust and t.bust == bust: score += 1
        if waist and t.waist == waist: score += 1
        if hips and t.hips == hips: score += 1
        if dress_size and t.dress_size == dress_size: score += 1
        if shoe_size and t.shoe_size == shoe_size: score += 1
        if job_type:
            j_types = t.job_types
            if isinstance(j_types, list):
                if any(matches(job_type, jt) for jt in j_types): score += 1
            elif matches(job_type, j_types): score += 1
                
        if budget is not None and t.budget_tier is not None:
            if t.budget_tier <= budget: score += 1
        if shoot_dates and t.availability:
            avail_str = str(t.availability)
            if any(d in avail_str for d in shoot_dates if d):
                score += 1
        
        if score > 0:
            scored_talents.append((score, t))

    scored_talents.sort(key=lambda x: x[0], reverse=True)
    top_results = [item[1] for item in scored_talents[:10]]
    
    result_list = []
    for t in top_results:
        photos = t.photos
        if isinstance(photos, list):
            photos = photos[0] if photos else None
        
        availability_list = []
        if t.availability:
            if isinstance(t.availability, str):
                availability_list = [d.strip() for d in t.availability.split(",") if d.strip()]
            elif isinstance(t.availability, list):
                availability_list = t.availability

        result_list.append({
            "id": t.id,
            "photos": photos,
            "availability": availability_list,
            "name": t.name,
            "height": t.height,
            "bust": t.bust,
            "waist": t.waist,
            "hips": t.hips,
            "dress_size": t.dress_size,
            "shoe_size": t.shoe_size,
            "hair": t.hair,
            "hair_type": t.hair_type,
            "eyes": t.eyes,
            "skin": t.skin_color,
            "agent_name": t.agent_name,
            "budget_tier": t.budget_tier,
        })
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
            result = generate_casting.invoke(tool_call["args"])
            
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
