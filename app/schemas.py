from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import date, datetime
from decimal import Decimal

class OptionalDetails(BaseModel):
    location: Optional[str] = None
    shoot_date: Optional[List[str]] = None

class TalentResponse(BaseModel):
    talent_id: int 
    images: List[str] = []
    is_active: bool
    name: str
    role: Optional[str] = None
    # added_by_agent_id: int
    agent_name: str
    date_of_birth: Optional[date] = None
    gender: str
    height: Optional[Decimal] = None
    bust: Optional[Decimal] = None
    waist: Optional[Decimal] = None
    hips: Optional[Decimal] = None
    shoe_size: Optional[str] = None
    dress_size: Optional[str] = None
    eye_color: str
    hair_type: str
    hair_color: str
    skin_color: str
    location: str
    continent: str
    country: str
    #available_date: Optional[date] = None add again

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    location: Optional[str] = None
    shoot_date: Optional[List[str]] = Field(None, alias="shoot_dates")
    budget: Optional[str] = Field(None, alias="budget_range")
    job_type: Optional[str] = None
    title:  Optional[str] = None
    description: Optional[str] = None
    save_as_draft: bool = False
    generate_job: bool = False

    class Config:
        populate_by_name = True

class SessionRequest(BaseModel):
    session_id: str
    
class GenerateCastingRequest(BaseModel):
    session_id: str

class SaveTalentRequest(BaseModel):
    session_id: str
    talent_id: int

class BookTalentRequest(BaseModel):
    session_id: Optional[str] = None
    talent_id: int

class ChatMessageResponse(BaseModel):
    sender: str
    content: str
    timestamp: datetime

    class Config:
        from_attributes = True

class JobResponse(BaseModel):
    job_id: int
    job_created_by_id: int
    session_id: Optional[str] = None
    status: str
    title: str
    job_type: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    budget: Optional[str] = None
    applicants_count: int
    shortlisted_count: int
    selftapes_count: int
    created_at: datetime

    class Config:
        from_attributes = True

class JobResultResponse(BaseModel):
    job_id: int
    job_created_by_id: int
    session_id: Optional[str] = None
    status: str
    title: str
    job_type: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    created_at: datetime
    applicants_count: int
    shortlisted_count: int
    selftapes_count: int
    shoot_date: Optional[str] = None
    suggested_talents: List[TalentResponse] = []
    messages: List[ChatMessageResponse] = []

    class Config:
        from_attributes = True

class DraftResponse(BaseModel):
    draft_id: int
    user_id: int
    session_id: str
    # phase: str
    saved_filters: Dict[str, Any] = {}
    last_updated: Optional[datetime] = None
    messages: List[ChatMessageResponse] = []

    class Config:
        from_attributes = True

class DraftsSavedFilters(BaseModel):
    job_type: Optional[str] = Field(None, alias="Job type")
    message: Optional[str] = Field(None, alias="Message")

    class Config:
        populate_by_name = True

class UserDraftResponse(BaseModel):
    draft_id: int
    user_id: int
    session_id: str
    saved_filters: DraftsSavedFilters
    updated: str = Field(..., alias="Updated")
    last_updated: Optional[datetime] = None

    class Config:
        from_attributes = True
        populate_by_name = True

class ContinueDraftResponse(BaseModel):
    session_id: str
    messages: List[ChatMessageResponse] = []
    saved_filters: Dict[str, Any] = {}

    class Config:
        from_attributes = True

class ConversationResponse(BaseModel):
    text: str

class PaginationResponse(BaseModel):    
    total_results: int
    page: int
    per_page: int
    has_next: bool

class TalentDataResponse(BaseModel):
    talents: List[TalentResponse] = []

class WrappedChatResponse(BaseModel):
    session_id: str
    timestamp: str
    conversation: str
    pagination: Optional[PaginationResponse] = None
    data: Optional[TalentDataResponse] = None
    generated_job: Optional[JobResponse] = None

class ChatSessionResponse(BaseModel):
    session_id: str
    user_id: int
    created_at: datetime
    messages: List[ChatMessageResponse] = []

    class Config:
        from_attributes = True
