from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated, Optional
from fastapi import Request, HTTPException, status
from database import SessionLocal, Talent
from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL
from datetime import datetime, date
import json
import time
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

class RateLimiter:
    def __init__(self, limit: int, window: int, error_msg: str):
        self.limit = limit
        self.window = window
        self.error_msg = error_msg
        self.clients = defaultdict(list)

    async def __call__(self, request: Request):
        client_ip = request.client.host
        now = time.time()
        self.clients[client_ip] = [t for t in self.clients[client_ip] if now - t < self.window]
        if len(self.clients[client_ip]) >= self.limit:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=self.error_msg)
        self.clients[client_ip].append(now)
        
class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    filters: Dict[str, Any] = {}
    found_talents: List[Dict] = []

class GeneratedJobInfo(BaseModel):
    title: str = Field(..., description="A concise and catchy title for the casting job.")
    description: str = Field(..., description="A one-line summary of the job description.")

class ExtractedFilters(BaseModel):
    location: Optional[str] = Field(None, description="Location")
    continent: Optional[str] = Field(None, description="Continent")
    country: Optional[str] = Field(None, description="Country")
    gender: Optional[str] = Field(None, description="Gender")
    hair_color: Optional[str] = Field(None, description="Hair Color")
    eye_color: Optional[str] = Field(None, description="Eye Color")
    skin_color: Optional[str] = Field(None, description="Skin Color")
    shoot_date: Optional[List[str]] = Field(None, description="Shoot Dates")
    hair_type: Optional[str] = Field(None, description="Hair Type")
    height: Optional[str] = Field(None, description="Height")
    bust: Optional[str] = Field(None, description="Bust")
    waist: Optional[str] = Field(None, description="Waist")
    hips: Optional[str] = Field(None, description="Hips")
    shoe_size: Optional[str] = Field(None, description="Shoe Size")
    dress_size: Optional[str] = Field(None, description="Dress Size")
    budget: Optional[str] = Field(None, description="Budget")
    job_type: Optional[str] = Field(None, description="Job Type")

def extract_information(user_input: str, current_filters: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts casting filters from user input using a lightweight LLM call."""
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    structured_llm = llm.with_structured_output(ExtractedFilters)
    
    prompt = f"""
    You are a Casting Assistant. Extract casting requirements from the user's message.
    Current known info: {current_filters}
    User message: "{user_input}"
    
    Return ONLY the fields that are explicitly mentioned or updated in the user message.
    """
    try:
        result = structured_llm.invoke(prompt)
        return {k: v for k, v in result.dict().items() if v is not None}
    except Exception as e:
        return {}

def generate_job_details_from_messages(messages: List[str]) -> GeneratedJobInfo:
    """Generates a job title and description from a list of messages."""
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)
    structured_llm = llm.with_structured_output(GeneratedJobInfo)
    
    conversation_summary = "\n".join(messages)
    
    prompt = f"""
    Based on the following conversation snippets, generate a concise job title and a one-line job description for a casting call.

    Conversation:
    ---
    {conversation_summary}
    ---

    Example Output:
    Title: Runway Model Show
    Description: Seeking experienced runway models for a high-fashion event in Los Angeles.

    Generate the title and description.
    NOTE: Never add any physical or facial features in Title and Description. You may add gender or job type. 
    """
    
    try:
        result = structured_llm.invoke(prompt)
        return result
    except Exception as e:
        return GeneratedJobInfo(title="Casting Call", description="Casting for a new project.")

def generate_ask_response(missing_fields: List[str]) -> str:
    """Generates a polite question to ask for missing mandatory fields."""
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.6, api_key=OPENAI_API_KEY)
    prompt = f"""
    You are a Casting Assistant helping a user find talent. You need to collect the following missing criteria for the talent search: {', '.join(missing_fields)}.
    Politely ask the user for these criteria about the talent they are looking for.
    For example, if you need 'gender', ask 'What gender are you looking for in the talent?'.
    Keep the question short, professional, and focused on the talent search.
    """
    return llm.invoke(prompt).content

@tool
def generate_casting(location: str = None, continent: str = None, country: str = None,
                     gender: str = None, hair_color: str = None, eye_color: str = None, skin_color: str = None,
                     shoot_date: List[str] = None, hair_type: str = None,
                     height: str = None, bust: str = None, waist: str = None, hips: str = None, budget: str = None, job_type: str = None,
                     shoe_size: str = None, dress_size: str = None):
    """
    Search for talent. Returns ranked recommendations based on matched criteria.
    """
    db = SessionLocal()
    try:
        talents = db.query(Talent).outerjoin(Talent.agent).filter(Talent.is_active == True).all()
        
        scored_talents = []
        
        for t in talents:
            score = 0
            
            def matches(value, target):
                if not value or not target:
                    return False
                return str(value).lower() in str(target).lower()

            if location:
                if matches(location, t.location) or matches(location, t.country) or matches(location, t.continent):
                    score += 1
            if continent and matches(continent, t.continent): score += 1
            if country and matches(country, t.country): score += 1
            
            if gender:
                g_input = gender.lower()
                if g_input == "man": g_input = "male"
                elif g_input == "woman": g_input = "female"
                elif g_input in ["non-binary", "non binary"]: g_input = "nonbinary"
                elif g_input in ["non-binary", "non binary"]: g_input = "transgender"

                if not t.gender or t.gender.lower() != g_input:
                    continue
                score += 1
            
            if hair_color and matches(hair_color, t.hair_colour): score += 1
            if hair_type and matches(hair_type, t.hair_type): score += 1
            if eye_color and matches(eye_color, t.eye_colour): score += 1
            if skin_color and matches(skin_color, t.skin_color): score += 1

            try:
                if height and t.height is not None and float(t.height) == float(height): score += 1
            except (ValueError, TypeError): pass
            try:
                if bust and t.bust is not None and float(t.bust) == float(bust): score += 1
            except (ValueError, TypeError): pass
            try:
                if waist and t.waist is not None and float(t.waist) == float(waist): score += 1
            except (ValueError, TypeError): pass
            try:
                if hips and t.hips is not None and float(t.hips) == float(hips): score += 1
            except (ValueError, TypeError): pass
            try:
                if shoe_size and t.shoe_size is not None and int(t.shoe_size) == int(shoe_size): score += 1
            except (ValueError, TypeError): pass
            try:
                if dress_size and t.dress_size is not None and int(t.dress_size) == int(dress_size): score += 1
            except (ValueError, TypeError): pass

# add it again after availavle date added on database
            # if shoot_date and t.available_date:
            #     try:
            #         user_dates = [datetime.strptime(d.strip(), '%Y-%m-%d').date() for d in shoot_date if d]
            #         if t.available_date in user_dates:
            #             score += 1
            #     except ValueError:
            #         pass
            
            if score > 0:
                scored_talents.append((score, t))

        scored_talents.sort(key=lambda x: x[0], reverse=True)
        top_results = [item[1] for item in scored_talents[:100]]
        total_results = len(scored_talents)
        
        result_list = []
        for t in top_results:
            result_list.append({
                "talent_id": t.talent_id,
                "agent_name": t.agent.full_name if t.agent else "Unknown",
                "name": t.name,
                "date_of_birth": t.date_of_birth,
                "gender": t.gender,
                "height": t.height,
                "bust": t.bust,
                "waist": t.waist,
                "hips": t.hips,
                "shoe_size": t.shoe_size,
                "dress_size": t.dress_size,
                "eye_color": t.eye_colour,
                "hair_type": t.hair_type,
                "hair_color": t.hair_colour,
                "skin_color": t.skin_color,
                "location": t.location,
                "continent": t.continent,
                "country": t.country,
                "is_active": t.is_active,
                # "available_date": t.available_date,
                "images": [f"/media/{img.image}" for img in sorted(t.images, key=lambda x: x.image_id)[:1]] if t.images else [],
            })
        
        return {
            "talents": result_list,
            "total_results": total_results
        }
    finally:
        db.close()

def reasoner_node(state: AgentState):
    SYSTEM_PROMPT = """
    You are an Elite Casting Director helping a user find talent. Your primary role is to use the 'generate_casting' tool once all mandatory criteria are met.
    All 6 mandatory fields (Location, Shoot Date, Budget, Job Type, Gender, Skin Color) have been collected.
    Current search criteria: {filters}

    Your task now is to refine the search.

    Rules:
    1. Ask for missing mandatory fields first.
    2. Once mandatory fields are collected, suggest appearance filters (Eye Color, Hair Color) if not already provided.
    3. If the user provides appearance details, call 'generate_casting'.
    4. If the user declines to provide more details, call 'generate_casting'.
    5. Do NOT call 'generate_casting' until the 6 mandatory fields are present.

    Be concise."""
    
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
                content=f"Search completed. Found {result['total_results']} talents. The full list has been sent to the user interface.",
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

def time_ago(dt: Optional[datetime]) -> str:
    if not dt:
        return "never"

    if dt.tzinfo:
        now = datetime.now(dt.tzinfo)
    else:
        now = datetime.now()

    delta = now - dt

    if delta.days > 0:
        return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
    
    hours = delta.seconds // 3600
    if hours > 0:
        return f"{hours} hour{'s' if hours > 1 else ''} ago"

    minutes = delta.seconds // 60
    if minutes > 0:
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    
    return "just now"

def parse_budget(budget_str):
    if not budget_str: return None, None
    
    clean_str = str(budget_str).replace(',', '').replace('$', '').replace('£', '').replace('€', '')
    nums = re.findall(r'\d+(?:\.\d+)?', clean_str)
    
    if not nums: return None, None
    
    try:
        vals = [Decimal(n) for n in nums]
    except InvalidOperation:
        return None, None

    if len(vals) == 1:
        return vals[0], vals[0]
    
    v1, v2 = vals[0], vals[1]
    return min(v1, v2), max(v1, v2)
