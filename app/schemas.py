from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class OptionalDetails(BaseModel):
    location: Optional[str] = None
    shoot_dates: Optional[List[str]] = None
    budget_range: Optional[int] = None
    job_type: Optional[str] = None

class TalentResponse(BaseModel):
    id: str
    name: Optional[str] = None
    availability: Optional[List[str]] = None
    photos: Optional[str] = None
    height: Optional[str] = None
    bust: Optional[str] = None
    waist: Optional[str] = None
    hips: Optional[str] = None
    dress_size: Optional[str] = None
    shoe_size: Optional[str] = None
    hair: Optional[str] = None
    hair_type: Optional[str] = None
    eyes: Optional[str] = None
    skin: Optional[str] = None
    agent_name: Optional[str] = None
    budget_tier: Optional[int] = None
    actions: Dict[str, str] = Field(
        default_factory=lambda: {
            # "save": "POST /save-talent",
            # "book": "POST /book-talent",
            # "calendar": "GET /talent/{id}/calendar",
            # "selftape": "POST /talent/{id}/selftape",
            # "request_virtual_casting": "GET /talent/{id}/request_virtual_casting",
            # "request_polas": "POST /talent/{id}/polas"
        }
    )

class ChatRequest(BaseModel):
    user_id: str
    message: str
    location: Optional[str] = None
    shoot_dates: Optional[List[str]] = None
    budget_range: Optional[int] = None
    job_type: Optional[str] = None
    save_as_draft: bool = False

class UserRequest(BaseModel):
    user_id: str

class SessionRequest(BaseModel):
    user_id: str
    session_id: str
    
class DraftRequest(BaseModel):
    id: int
    user_id: str
    
class DraftUserRequest(BaseModel):
    user_id: str

class ChatResponse(BaseModel):
    session_id: str
    response_text: str
    suggested_talents: List[TalentResponse] = []

# class SaveTalentRequest(BaseModel):
#     user_id: str
#     session_id: str
#     talent_id: str

# class BookTalentRequest(BaseModel):
#     user_id: str
#     session_id: Optional[str] = None
#     talent_id: str

class ChatMessageResponse(BaseModel):
    sender: str
    content: str
    timestamp: str

    class Config:
        orm_mode = True

class DraftResponse(BaseModel):
    id: int
    user_id: str
    session_id: str
    phase: str
    saved_filters: Dict[str, Any] = {}
    last_updated: str

    class Config:
        orm_mode = True

class ChatSessionResponse(BaseModel):
    id: str
    user_id: str
    created_at: str
    messages: List[ChatMessageResponse] = []

    class Config:
        orm_mode = True
