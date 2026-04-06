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
    agent_id: Optional[int] = None
    agent_name: Optional[str] = None
    date_of_birth: Optional[date] = None
    gender: Optional[str] = None
    height: Optional[Decimal] = None
    bust: Optional[Decimal] = None
    waist: Optional[Decimal] = None
    hips: Optional[Decimal] = None
    shoe_size: Optional[str] = None
    dress_size: Optional[str] = None
    eye_color: Optional[str] = None
    hair_type: Optional[str] = None
    hair_color: Optional[str] = None
    skin_color: Optional[str] = None
    location: Optional[str] = None
    continent: Optional[str] = None
    country: Optional[str] = None
    available_dates: List[date] = []
    status: Optional[str] = None
    tapes: List[str] = []
    polas: List[str] = []

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    location: Optional[str] = None
    shoot_date: Optional[List[str]] = Field(None, alias="shoot_dates")
    budget: Optional[str] = Field(None, alias="budget_range")
    job_type: Optional[str] = None
    limit: Optional[int] = None
    title:  Optional[str] = None
    description: Optional[str] = None
    save_as_draft: bool = False
    generate_job: bool = False

    class Config:
        populate_by_name = True

class SessionRequest(BaseModel):
    session_id: str
    
class RequestTalentJobRequest(BaseModel):
    job_id: int
    talent_id: int

class GenerateCastingRequest(BaseModel):
    session_id: str

class ShortlistTalentRequest(BaseModel):
    job_id: int
    talent_id: int
    session_id: Optional[str] = None

class BookTalentRequest(BaseModel):
    job_id: int
    talent_id: int
    session_id: Optional[str] = None

class SelfTapeStatusAction(BaseModel):
    job_id: int
    talent_id: int
    status: str  # 'accepted' or 'rejected'

class PolaStatusAction(BaseModel):
    job_id: int
    talent_id: int
    status: str  # 'accepted' or 'rejected'

class PolaUploadPageResponse(BaseModel):
    talent_name: str
    talent_role: Optional[str] = None
    job_title: str
    job_budget: Optional[str] = None
    timeline: Optional[str] = None # Shoot dates
    status: str
    existing_images: List[str] = []

class SelfTapeUploadRequest(BaseModel):
    job_id: int
    talent_id: int
    tape_urls: List[str]

class SelfTapeUploadPageResponse(BaseModel):
    talent_name: str
    talent_role: Optional[str] = None
    job_title: str
    job_budget: Optional[str] = None
    timeline: Optional[str] = None # Shoot dates
    status: str
    existing_tapes: List[str] = []

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
    ecastings_count: int
    polas_count: int
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
    ecastings_count: int
    polas_count: int
    shoot_date: Optional[str] = None
    suggested_talents: List[TalentResponse] = []
    requested_selftapes: List[TalentResponse] = []
    requested_ecastings: List[TalentResponse] = []
    requested_polas: List[TalentResponse] = []
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
    saved_filters: Dict[str, Any] = {}

    class Config:
        from_attributes = True

class TalentPreview(BaseModel):
    talent_id: str
    profile_image_url: Optional[str] = None

class ShortlistSummaryItem(BaseModel):
    job_id: str
    job_title: str
    talent_count: int
    time_remaining_hours: int
    preview_talents: List[TalentPreview]
    extra_talent_count: int

class SummaryPagination(BaseModel):
    page: int
    limit: int
    total: int

class ShortlistSummaryResponse(BaseModel):
    shortlists: List[ShortlistSummaryItem]
    pagination: SummaryPagination
