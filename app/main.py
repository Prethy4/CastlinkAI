import math
import os
import shutil
import uuid
from dotenv import load_dotenv
load_dotenv(override=True)

from datetime import datetime, timezone, timedelta, date
from fastapi import FastAPI, HTTPException, Depends, Query, Request, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.orm.attributes import flag_modified
from langchain_core.messages import HumanMessage, AIMessage
from app.database import init_db, get_db, ChatSession, ChatMessage, Draft, Job, JobAIResult, UserAuth, Talent, ShortlistedTalent, Booking, SelfTapeRequest, SelfTapeLink, PolaRequest, PolaLink, TalentAvailableDate
from app.schemas import TalentResponse, ChatSessionResponse, DraftResponse, ChatRequest, JobResponse, ContinueDraftResponse, ChatMessageResponse, JobResultResponse, WrappedChatResponse, PaginationResponse, TalentDataResponse, UserDraftResponse, DraftsSavedFilters, RequestTalentJobRequest, ShortlistTalentRequest, BookTalentRequest, ShortlistSummaryResponse, ShortlistSummaryItem, TalentPreview, SummaryPagination, SelfTapeStatusAction, SelfTapeUploadRequest, SelfTapeUploadPageResponse, PolaStatusAction, PolaUploadPageResponse, GenerateJobRequest
from app.services import app_graph, extract_information, generate_ask_response, CustomEncoder, RateLimiter, time_ago, parse_budget, generate_job_details_from_messages
from typing import List, Optional
import json
from fastapi.middleware.cors import CORSMiddleware

# Initialize database
init_db()
from app.auth import get_current_user

app = FastAPI(title="AI-Powered Casting")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the media directory to serve static files (videos and images)
app.mount("/media", StaticFiles(directory="media"), name="media")

limiter = RateLimiter(limit=20, window=60, error_msg="Something went wrong. Please contact the support.")

MANDATORY_FIELDS = ["location", "shoot_date", "budget", "job_type", "gender", "skin_color"]

##########-------Exception Handlers-------##########

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status_code": exc.status_code,
            "status_message": exc.detail
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "status_code": 422,
            "status_message": "Validation Error",
            "details": exc.errors()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status_code": 500,
            "status_message": "An unexpected error occurred. Please contact support."
        }
    )

##########-------health check-------##########

@app.get("/health")
async def health_check():
    return {
        "status_code": 200,
        "status_message": "Success",
        "message": "server running"
    }

@app.post("/api/chat", dependencies=[Depends(limiter)])
async def send_message(
    request: ChatRequest,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user)
    ):
    """Chat services for conversation with the chatbot"""
    try:
        user = db.query(UserAuth).filter(UserAuth.user_id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        message = request.message
        session_id = request.session_id.strip() if request.session_id and isinstance(request.session_id, str) and request.session_id.strip() else None
        save_as_draft = request.save_as_draft

        chat_session = None

        if session_id:
            chat_session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            
            if chat_session:
                if chat_session.user_id != user_id:
                    chat_session = ChatSession(user_id=user_id)
                    db.add(chat_session)
                    db.commit()
                    
            else:
                chat_session = ChatSession(user_id=user_id, session_id=session_id)
                db.add(chat_session)
                db.commit()
        else:
            chat_session = ChatSession(user_id=user_id)
            db.add(chat_session)
            db.commit()

        # --- Initialize and Populate Filters before creating user_msg ---
        filters = {}
        draft = db.query(Draft).filter(Draft.session_id == chat_session.session_id).first()
        if not draft:
            draft = Draft(session_id=chat_session.session_id, user_id=user_id)
            db.add(draft)
            db.flush()

        if draft.saved_filters:
            filters.update(draft.saved_filters)

        if not filters.get("location") and draft.location: filters["location"] = draft.location
        if not filters.get("budget") and draft.budget: filters["budget"] = draft.budget
        if not filters.get("job_type") and draft.job_type: filters["job_type"] = draft.job_type
        if not filters.get("shoot_date") and draft.shoot_date:
            filters["shoot_date"] = [d.strip() for d in draft.shoot_date.split(",") if d.strip()]
        if not filters.get("title") and draft.title: filters["title"] = draft.title
        if not filters.get("description") and draft.description: filters["description"] = draft.description

        user_msg = ChatMessage(
            session_id=chat_session.session_id, 
            sender="user", 
            content=message,
            filters=None
        )
        db.add(user_msg)

        db.commit()

        past_messages = db.query(ChatMessage).filter(ChatMessage.session_id == chat_session.session_id).order_by(ChatMessage.message_id.desc()).limit(20).all()
        past_messages.reverse()
        msgs = []
        for m in past_messages:
            if m.sender == "user":
                msgs.append(HumanMessage(content=m.content))
            else:
                msgs.append(AIMessage(content=m.content))

        existing_job = db.query(Job).filter(Job.session_id == chat_session.session_id).order_by(Job.created_at.desc()).first()

        if not filters.get("title"):
            filters["title"] = (existing_job.title if existing_job else None) or draft.title or filters.get("title")
        if not filters.get("description"):
            filters["description"] = (existing_job.description if existing_job else None) or draft.description or filters.get("description")

        # Update Title/Description if provided else generate
        if request.title: 
            filters['title'] = request.title
            if existing_job: existing_job.title = request.title
        if request.description: 
            filters['description'] = request.description
            if existing_job: existing_job.description = request.description

        # --- Save as Draft Logic ---
        if save_as_draft:
            draft.saved_filters = filters
            draft.phase = "saved"
            db.commit()
            return {
                "status_code": 200,
                "status_message": "Draft saved successfully.",
                "session_id": chat_session.session_id
            }

        # --- Extract information from the message text (if provided) ---
        if message.strip():
            extracted_updates = extract_information(message, filters)
            filters.update(extracted_updates)

        # --- Update filters with current request values (priority) ---

        if request.location is not None: filters['location'] = request.location
        if request.budget is not None: filters['budget'] = request.budget
        if request.job_type is not None: filters['job_type'] = request.job_type
        if request.gender is not None: filters['gender'] = request.gender
        if request.skin_color is not None: filters['skin_color'] = request.skin_color
        if request.role is not None: filters['role'] = request.role
        if request.limit is not None: filters['limit'] = request.limit
        if request.title is not None: filters['title'] = request.title
        if request.description is not None: filters['description'] = request.description

        # Handle shoot_date 
        if request.shoot_date is not None:
            cleaned_dates = []
            for d in request.shoot_date:
                if isinstance(d, str):
                    parts = [dt.strip() for dt in d.split(",") if dt.strip()] if "," in d else [d.strip()]
                    for dt_str in parts:
                        if dt_str: cleaned_dates.append(dt_str)
            if cleaned_dates:
                filters['shoot_date'] = list(dict.fromkeys(cleaned_dates))

        user_msg.location = filters.get("location")
        user_msg.budget = filters.get("budget")
        user_msg.job_type = filters.get("job_type")
        s_dates_msg = filters.get("shoot_date")
        user_msg.shoot_date = ", ".join(s_dates_msg) if isinstance(s_dates_msg, list) else s_dates_msg

        if not filters.get("title") or not filters.get("description"):
            context_contents = [m.content for m in msgs] + [message]
            generated_info = generate_job_details_from_messages(context_contents)
            filters["title"] = filters.get("title") or generated_info.title
            filters["description"] = filters.get("description") or generated_info.description
            
            draft.title = filters.get("title") or draft.title
            draft.description = filters.get("description") or draft.description
            
            if existing_job:
                if not existing_job.title: existing_job.title = filters["title"]
                if not existing_job.description: existing_job.description = filters["description"]

        # ---Check for mandatory fields--- #
        missing_keys = [k for k in MANDATORY_FIELDS if not filters.get(k)]

        final_state = {}
        response_content = ""

        if missing_keys:
            is_initial = len(msgs) <= 1 or any(greet in message.lower() for greet in ["hi", "hello", "hey", "greetings"])
            response_content = generate_ask_response(missing_keys, message, is_initial)
            final_state = {"filters": filters, "messages": [AIMessage(content=response_content)]}
        else:
            inputs = {"messages": msgs, "filters": filters}
            final_state = app_graph.invoke(inputs, config={"recursion_limit": 20})
            if final_state.get('messages'):
                last_msg = final_state['messages'][-1]
                response_content = last_msg.content or "How else can I help you with your search?"
            else:
                response_content = "I'm processing your search. Could you provide more details?"
        
        # --- Persist Session State (Draft) --- #
        current_filters = final_state.get('filters', {})
        if "title" not in current_filters and filters.get("title"): current_filters["title"] = filters["title"]
        if "description" not in current_filters and filters.get("description"): current_filters["description"] = filters["description"]

        suggested_talents_list = []
        total_results = 0
        search_performed = False
        for msg in final_state.get('messages', []):
            if hasattr(msg, 'name') and msg.name == 'generate_casting':
                search_performed = True
                if hasattr(msg, 'artifact') and msg.artifact:
                    suggested_talents_list = msg.artifact.get('talents', [])
                    total_results = msg.artifact.get('total_results', 0)

        if search_performed:
            current_filters['suggested_count'] = len(suggested_talents_list)
            current_filters['suggested_talents_list'] = suggested_talents_list
            current_filters['total_results'] = total_results
            draft.phase = "generated" 

            jobs_in_session = db.query(Job).filter(Job.session_id == chat_session.session_id).all()
            for job in jobs_in_session:
                job.applicants_count = total_results
                db.add(job)

        current_phase = "READY_TO_GENERATE" if all(k in current_filters for k in MANDATORY_FIELDS) else "COLLECT_MANDATORY"

        draft.title = current_filters.get("title") or draft.title
        draft.description = current_filters.get("description") or draft.description
        draft.location = current_filters.get("location")
        draft.budget = current_filters.get("budget")
        draft.job_type = current_filters.get("job_type")
        s_dates = current_filters.get("shoot_date")
        if isinstance(s_dates, list):
            draft.shoot_date = ", ".join(s_dates)
        else:
            draft.shoot_date = s_dates
        
        current_filters['last_updated_timestamp'] = datetime.now(timezone.utc).isoformat()
        draft.saved_filters = json.loads(json.dumps(current_filters, cls=CustomEncoder))
        db.commit()

        ai_msg = ChatMessage(
            session_id=chat_session.session_id, 
            sender="ai", 
            content=response_content,
            filters=json.loads(json.dumps(current_filters, cls=CustomEncoder)) if search_performed else None
        )
        db.add(ai_msg)
        db.commit()

        db.refresh(chat_session)
        response = WrappedChatResponse(
            status_code=200,
            status_message="Success",
            data=ChatSessionResponse.from_orm(chat_session)
        )
        return jsonable_encoder(response)

    except HTTPException:
        raise
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
        #raise HTTPException(status_code=500, detail="Something went wrong. Please try again later or contact the support.")

@app.post("/api/jobs/generate", dependencies=[Depends(limiter)])
async def generate_job_api(
    request: GenerateJobRequest,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user)
):
    """Explicitly generate a job from a chat session's collected data."""
    draft = db.query(Draft).filter(Draft.session_id == request.session_id, Draft.user_id == user_id).first()
    if not draft:
        raise HTTPException(status_code=404, detail="Session data or draft not found.")

    current_filters = draft.saved_filters or {}
    
    # --- Check Mandatory Fields ---
    missing_fields = [k for k in MANDATORY_FIELDS if not current_filters.get(k)]
    if missing_fields:
        missing_str = ", ".join(k.replace('_', ' ').title() for k in missing_fields)
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot generate job. Missing required details: {missing_str}."
        )

    # --- Logic for Title and Description ---
    job_title = request.title or current_filters.get("title") or draft.title
    job_description = request.description or current_filters.get("description") or draft.description

    if not job_title or not job_description:
        initial_messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == request.session_id
        ).order_by(ChatMessage.message_id.asc()).limit(10).all()
        message_contents = [m.content for m in initial_messages]
        
        if message_contents:
            generated_details = generate_job_details_from_messages(message_contents)
            job_title = job_title or generated_details.title
            job_description = job_description or generated_details.description

    job_title = job_title or "Casting Call"
    job_description = job_description or "Casting for a new project."

    current_filters['last_updated_timestamp'] = datetime.now(timezone.utc).isoformat()
    draft.title = job_title
    draft.description = job_description
    current_filters["title"] = job_title
    current_filters["description"] = job_description
    draft.saved_filters = json.loads(json.dumps(current_filters, cls=CustomEncoder))

    # --- Extract metadata for Job object ---
    d_location = current_filters.get("location")
    d_job_type = current_filters.get("job_type")
    d_budget = current_filters.get("budget")
    s_dates_raw = current_filters.get("shoot_date")
    d_shoot_date = ", ".join(s_dates_raw) if isinstance(s_dates_raw, list) else s_dates_raw
    
    st_list = current_filters.get('requested_selftapes', [])
    ec_list = current_filters.get('requested_ecastings', [])
    pl_list = current_filters.get('requested_polas', [])
    suggested_talents_list = current_filters.get('suggested_talents_list', [])
    total_applicants = current_filters.get('total_results', 0)
    
    budget_min, budget_max = parse_budget(d_budget)

    # --- Create Job ---
    new_job = Job(
        job_created_by_id=user_id, 
        session_id=request.session_id,
        title=job_title, 
        description=job_description, 
        location=d_location,
        budget_min=budget_min, 
        budget_max=budget_max, 
        job_type=d_job_type,
        status="active",
        applicants_count=total_applicants, 
        shortlisted_count=db.query(ShortlistedTalent).filter(ShortlistedTalent.session_id == request.session_id, ShortlistedTalent.job_id == None).count(),
        selftapes_count=len(st_list), 
        ecastings_count=len(ec_list), 
        polas_count=len(pl_list)
    )
    db.add(new_job)
    db.flush()

    # Link existing session data
    db.query(ShortlistedTalent).filter(ShortlistedTalent.session_id == request.session_id, ShortlistedTalent.job_id == None).update({"job_id": new_job.job_id}, synchronize_session=False)
    db.query(Booking).filter(Booking.session_id == request.session_id, Booking.job_id == None).update({"job_id": new_job.job_id}, synchronize_session=False)

    for t in st_list: db.add(SelfTapeRequest(job_id=new_job.job_id, talent_id=t['talent_id'], status=t.get('status', 'requested')))
    for t in pl_list: db.add(PolaRequest(job_id=new_job.job_id, talent_id=t['talent_id'], status=t.get('status', 'requested')))

    ai_result = JobAIResult(
        job_id=new_job.job_id, 
        suggested_talents=suggested_talents_list,
        shoot_date=d_shoot_date,
        requested_selftapes=st_list,
        requested_ecastings=ec_list,
        requested_polas=pl_list
    )
    db.add(ai_result)
    draft.phase = "generated"
    db.commit()

    detailed_data = await view_ai_result(new_job.job_id, user_id, db)
    detailed_data["status_message"] = f"Job '{new_job.title}' created successfully."
    return detailed_data

###########----------chat session-----------############

@app.get("/api/chat/sessions", dependencies=[Depends(limiter)])
async def get_sessions(
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
    ):
    """Get chat session of an user"""

    sessions = db.query(ChatSession).options(
        joinedload(ChatSession.messages),
        joinedload(ChatSession.draft)
    ).filter(ChatSession.user_id == user_id).all()

    if not sessions:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "status_code": 200,
        "status_message": "Success",
        "data": [jsonable_encoder(ChatSessionResponse.from_orm(s)) for s in sessions]
    }

############----------get chat session by id--------############ recheck

@app.get("/api/chat/session-id", dependencies=[Depends(limiter)])
async def get_session_details(
    session_id: str,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
    ):
    """Get chat session by session id"""

    session = db.query(ChatSession).options(
        joinedload(ChatSession.messages),
        joinedload(ChatSession.draft)
    ).filter(ChatSession.session_id == session_id, ChatSession.user_id == user_id).first()

    if not session:
        raise HTTPException(status_code=404, detail="Chat not found")

    db.refresh(session)
    response = WrappedChatResponse(
        status_code=200,
        status_message="Success",
        data=ChatSessionResponse.from_orm(session)
    )
    return jsonable_encoder(response)

# ###########----------delete chat-----------############
# @app.delete("/api/chat/delete-session-id", dependencies=[Depends(limiter)])
# async def delete_session(
#     request: SessionRequest, 
#     db: Session = Depends(get_db)
#     ):

#         session = db.query(ChatSession).filter(ChatSession.session_id == request.session_id).first()
#         if not session:
#             raise HTTPException(status_code=404, detail="Chat not found")
#         db.delete(session)
#         db.commit()
#         return {"status": "success", "message": "Chat deleted"}
    
#############----------retrives all draft state------------###############

@app.get("/api/chat/drafts", dependencies=[Depends(limiter)])
async def get_user_drafts(
    user_id: int = Depends(get_current_user),
    search: Optional[str] = Query(None, alias="search", description="Search drafts by job type"),
    db: Session = Depends(get_db)
    ):
    """Get all Draft states"""

    query = db.query(Draft).filter(
        Draft.user_id == user_id, 
        Draft.phase.in_(["saved", "generated", None])
    )
    all_drafts = query.all()
        
    drafts = []
    if search:
        search_lower = search.lower()
        for draft in all_drafts:
                
            if draft.job_type and search_lower in draft.job_type.lower():
                drafts.append(draft)
                continue
                
            if draft.saved_filters and isinstance(draft.saved_filters, dict):
                val = draft.saved_filters.get("job_type")
                if val and search_lower in str(val).lower():
                    drafts.append(draft)
    else:
        drafts = all_drafts

    response_drafts = []
    for draft in drafts:
        last_user_message = db.query(ChatMessage).filter(
            ChatMessage.session_id == draft.session_id,
            ChatMessage.sender == 'user'
        ).order_by(ChatMessage.message_id.desc()).first()

        message_content = last_user_message.content if last_user_message else None
            
        job_type = draft.job_type
        if not job_type and isinstance(draft.saved_filters, dict):
            job_type = draft.saved_filters.get('job_type')

        display_time = draft.last_updated or datetime.now(timezone.utc)

        saved_filters_response = DraftsSavedFilters(
            job_type=job_type,
            message=message_content,
            last_updated_timestamp=display_time.isoformat()
        )

        response_drafts.append(
            UserDraftResponse(
                draft_id=draft.draft_id,
                user_id=draft.user_id,
                session_id=draft.session_id,
                title=draft.title,
                description=draft.description,
                saved_filters=saved_filters_response,
                generate_job=draft.generate_job,
                Updated=time_ago(display_time),
                last_updated=display_time
            )
        )
                
    return {
        "status_code": 200,
        "status_message": "Success",
        "data": jsonable_encoder(response_drafts)
    }

#############----------retrives draft state------------###############

@app.get("/api/chat/draft-id", dependencies=[Depends(limiter)])
async def get_draft(
    draft_id: int,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
    ):
    """Retrives particular Draft"""
    
    draft = db.query(Draft).filter(Draft.draft_id == draft_id, Draft.user_id == user_id).first()
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    
    session = db.query(ChatSession).filter(ChatSession.session_id == draft.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Associated session not found")

    db.refresh(session)
    response = WrappedChatResponse(
        status_code=200,
        status_message="Success",
        data=ChatSessionResponse.from_orm(session)
    )
    return jsonable_encoder(response)

############--------continue draft------##############

@app.get("/api/chat/continue-draft-id", dependencies=[Depends(limiter)])
async def continue_draft(
    draft_id: int,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
    ):
    """Continue to an Draft"""
    
    draft = db.query(Draft).filter(Draft.draft_id == draft_id, Draft.user_id == user_id).first()
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    
    session = db.query(ChatSession).filter(ChatSession.session_id == draft.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Associated session not found")
    
    db.refresh(session)
    response = WrappedChatResponse(
        status_code=200,
        status_message="Success",
        data=ChatSessionResponse.from_orm(session)
    )
    return jsonable_encoder(response)

###########---------delete draft---------############

@app.delete("/api/chat/delete-draft-id", dependencies=[Depends(limiter)])
async def delete_draft(
    draft_id: int,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
    ):
    
    draft = db.query(Draft).filter(Draft.draft_id == draft_id, Draft.user_id == user_id).first()
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    
    db.delete(draft)
    db.commit()
    return {
        "status_code": 200, 
        "status_message": "Draft deleted"
    }

##########-------------get generated jobs-----------############

@app.get("/api/retrive-generated-jobs", dependencies=[Depends(limiter)])
async def get_user_jobs(
    user_id: int = Depends(get_current_user),
    search: Optional[str] = Query(None, description="Search by job type"),
    sort: Optional[str] = Query("all", description="Sort options: all, urgent, this week"),
    db: Session = Depends(get_db)
    ):
    """
    Return all jobs for a user.
    """
    query = db.query(Job).filter(Job.job_created_by_id == user_id)

    if search:
        query = query.filter(Job.job_type.ilike(f"%{search}%"))

    if sort and sort.lower() != 'all':
        now = datetime.now(timezone.utc)
        if sort.lower() == 'urgent':
            start_of_day = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            query = query.filter(Job.created_at >= start_of_day)
        elif sort.lower() == 'this week':
            week_ago = now - timedelta(days=7)
            query = query.filter(Job.created_at >= week_ago)

    query = query.order_by(Job.created_at.desc())
    
    return {
        "status_code": 200,
        "status_message": "Success",
        "data": [jsonable_encoder(JobResponse.from_orm(j)) for j in query.all()]
    }

############-----------view AI results------------###############

@app.get("/api/jobs/view-ai-result", dependencies=[Depends(limiter)])
async def view_ai_result(
    job_id: int,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    View the final AI result for a specific created job, including the list of suggested talents.
    """
    job = db.query(Job).filter(Job.job_id == job_id, Job.job_created_by_id == user_id).first()
    if not job:
        raise HTTPException(status_code=401, detail="User unauthorised or Job not found")

    db.refresh(job)
    ai_result = db.query(JobAIResult).filter(JobAIResult.job_id == job.job_id).first()
    suggested_talents = ai_result.suggested_talents if ai_result else []
    requested_selftapes_raw = ai_result.requested_selftapes if ai_result else []
    requested_ecastings_raw = ai_result.requested_ecastings if ai_result else []
    requested_polas_raw = ai_result.requested_polas if ai_result else []
    shoot_date = ai_result.shoot_date if ai_result else None

    messages = []
    if job.session_id:
        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == job.session_id,
            ChatMessage.timestamp <= job.created_at
        ).order_by(ChatMessage.message_id.asc()).all()

    # Fetch relational data for Self-Tapes
    st_data = db.query(SelfTapeRequest).options(joinedload(SelfTapeRequest.tapes)).filter(SelfTapeRequest.job_id == job_id).all()
    st_status_map = {r.talent_id: r.status for r in st_data}
    st_links_map = {r.talent_id: [t.tape_url for t in r.tapes] for r in st_data}

    # Fetch relational data for Polas
    pola_data = db.query(PolaRequest).options(joinedload(PolaRequest.images)).filter(PolaRequest.job_id == job_id).all()
    pola_status_map = {r.talent_id: r.status for r in pola_data}
    pola_links_map = {r.talent_id: [img.pola_url for img in r.images] for r in pola_data}

    def prepare_talent_list(raw_list, list_type):
        processed = []
        for t in (raw_list or []):
            talent_id = t.get('talent_id')
            
            if list_type == 'selftape':
                if talent_id in st_status_map: t['status'] = st_status_map[talent_id]
                if talent_id in st_links_map: t['tapes'] = st_links_map[talent_id]
                else: t['tapes'] = t.get('tapes', [])
                
            elif list_type == 'polas':
                if talent_id in pola_status_map: t['status'] = pola_status_map[talent_id]
                if talent_id in pola_links_map: t['polas'] = pola_links_map[talent_id]
                else: t['polas'] = t.get('polas', [])
            
            # For e-castings, snapshots are returned as-is
            processed.append(TalentResponse(**t))
        return processed

    # Universal standard format: merge all into one talents list
    all_talents_map = {}
    for t in (suggested_talents or []):
        all_talents_map[t.get('talent_id')] = TalentResponse(**t)
    for t in prepare_talent_list(requested_selftapes_raw, 'selftape'):
        all_talents_map[t.talent_id] = t
    for t in [TalentResponse(**t) for t in (requested_ecastings_raw or [])]:
        all_talents_map[t.talent_id] = t
    for t in prepare_talent_list(requested_polas_raw, 'polas'):
        all_talents_map[t.talent_id] = t

    final_talents = list(all_talents_map.values())
    total_results = len(final_talents)

    session = db.query(ChatSession).filter(ChatSession.session_id == job.session_id).first()
    
    # Construct filters manually to include the latest talent data (including status)
    updated_filters = session.saved_filters.copy() if session else {}
    updated_filters['suggested_talents_list'] = [jsonable_encoder(t) for t in final_talents]
    updated_filters['last_updated_timestamp'] = datetime.now(timezone.utc).isoformat()
    updated_filters['total_results'] = total_results

    response = WrappedChatResponse(
        status_code=200,
        status_message="Success",
        data=ChatSessionResponse(
            session_id=job.session_id,
            user_id=user_id,
            created_at=job.created_at,
            messages=[ChatMessageResponse.from_orm(m) for m in messages],
            generate_job=True
        )
    )
    return jsonable_encoder(response)

########---------request fot selftape--------############

@app.post("/api/jobs/request-selftape", dependencies=[Depends(limiter)])
async def request_selftape(
    request: RequestTalentJobRequest,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Endpoint to request a self-tape for a specific talent in a job."""
    job = None
    if request.job_id:
        job = db.query(Job).filter(Job.job_id == request.job_id, Job.job_created_by_id == user_id).first()
    elif request.session_id:
        job = db.query(Job).filter(Job.session_id == request.session_id, Job.job_created_by_id == user_id).order_by(Job.created_at.desc()).first()

    talent = db.query(Talent).filter(Talent.talent_id == request.talent_id).first()
    if not talent:
        raise HTTPException(status_code=404, detail="Talent not found")

    if not job:
        if not request.session_id:
            raise HTTPException(status_code=400, detail="Job not found and no session_id provided")
        draft = db.query(Draft).filter(Draft.session_id == request.session_id, Draft.user_id == user_id).first()
        if not draft:
            raise HTTPException(status_code=404, detail="Session not found to store request")

        filters = draft.saved_filters or {}
        selftapes_list = filters.get('requested_selftapes', [])
        if any(t.get('talent_id') == talent.talent_id for t in selftapes_list):
            raise HTTPException(status_code=400, detail="Selftape already added")

        talent_snapshot = {
            "talent_id": talent.talent_id,
            "job_id": None,
            "name": talent.name,
            "role": talent.role,
            "gender": talent.gender,
            "location": talent.location,
            "country": talent.country,
            "continent": talent.continent,
            "is_active": talent.is_active,
            "approval_status": talent.approval_status,
            "is_available": talent.is_available,
            "is_available_on_request": talent.is_available_on_request,
            "agent_id": talent.agent_id,
            "agent_name": talent.agent.full_name if talent.agent else "Unknown",
            "images": [f"/media/{img.image}" for img in sorted(talent.images, key=lambda x: x.image_id)] if talent.images else [],
            "eye_color": talent.eye_colour,
            "hair_type": talent.hair_type,
            "hair_color": talent.hair_colour,
            "skin_color": talent.skin_color,
            "height": talent.height, "bust": talent.bust, "waist": talent.waist, "hips": talent.hips,
            "shoe_size": talent.shoe_size, "dress_size": talent.dress_size,
            "available_dates": [ad.available_date for ad in talent.available_dates if ad.is_active],
            "status": "requested",
            "tapes": []
        }
        filters['requested_selftapes'] = selftapes_list + [talent_snapshot]
        filters['last_updated_timestamp'] = datetime.now(timezone.utc).isoformat()
        draft.saved_filters = filters
        flag_modified(draft, "saved_filters")
        db.commit()
        return {
            "status_code": 200, 
            "status_message": f"Self-tape requested for {talent.name} (Pending job generation)"
        }

    db.refresh(job)

    ai_result = db.query(JobAIResult).filter(JobAIResult.job_id == job.job_id).first()
    if not ai_result:
        ai_result = JobAIResult(job_id=job.job_id, requested_selftapes=[])
        db.add(ai_result)
    
    selftapes_list = ai_result.requested_selftapes or []

    if any(t.get('talent_id') == talent.talent_id for t in selftapes_list):
        raise HTTPException(status_code=400, detail="Self-tape already added")

    job.selftapes_count = (job.selftapes_count or 0) + 1

    talent_snapshot = {
        "talent_id": talent.talent_id,
        "job_id": job.job_id,
        "name": talent.name,
        "role": talent.role,
        "gender": talent.gender,
        "location": talent.location,
        "country": talent.country,
        "continent": talent.continent,
        "is_active": talent.is_active,
        "approval_status": talent.approval_status,
        "is_available": talent.is_available,
        "is_available_on_request": talent.is_available_on_request,
        "agent_id": talent.agent_id,
        "agent_name": talent.agent.full_name if talent.agent else "Unknown",
        "images": [f"/media/{img.image}" for img in sorted(talent.images, key=lambda x: x.image_id)] if talent.images else [],
        "eye_color": talent.eye_colour,
        "hair_type": talent.hair_type,
        "hair_color": talent.hair_colour,
        "skin_color": talent.skin_color,
        "height": talent.height, "bust": talent.bust, "waist": talent.waist, "hips": talent.hips,
        "shoe_size": talent.shoe_size, "dress_size": talent.dress_size,
        "available_dates": [ad.available_date for ad in talent.available_dates if ad.is_active],
        "status": "requested",
        "tapes": []
    }

    # Create the relational record with the dedicated status column
    new_st_request = SelfTapeRequest(
        job_id=job.job_id,
        talent_id=talent.talent_id,
        status="requested"
    )
    db.add(new_st_request)
    
    ai_result.requested_selftapes = list(selftapes_list) + [talent_snapshot]
 
    # Update session draft timestamp for consistency in chat history
    if job.session and job.session.draft:
        job.session.draft.saved_filters['last_updated_timestamp'] = datetime.now(timezone.utc).isoformat()
        flag_modified(job.session.draft, "saved_filters")

    db.commit()
    return {
        "status_code": 200, 
        "status_message": f"Self-tape requested for {talent.name}"
    }

#########--------request for e-casting--------##########

@app.post("/api/jobs/request-ecasting", dependencies=[Depends(limiter)])
async def request_ecasting(
    request: RequestTalentJobRequest,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Endpoint to request an e-casting for a specific talent in a job."""
    job = None
    if request.job_id:
        job = db.query(Job).filter(Job.job_id == request.job_id, Job.job_created_by_id == user_id).first()
    elif request.session_id:
        job = db.query(Job).filter(Job.session_id == request.session_id, Job.job_created_by_id == user_id).order_by(Job.created_at.desc()).first()

    talent = db.query(Talent).filter(Talent.talent_id == request.talent_id).first()
    if not talent:
        raise HTTPException(status_code=404, detail="Talent not found")

    if not job:
        if not request.session_id:
            raise HTTPException(status_code=400, detail="Job not found and no session_id provided")
        draft = db.query(Draft).filter(Draft.session_id == request.session_id, Draft.user_id == user_id).first()
        if not draft:
            raise HTTPException(status_code=404, detail="Session not found to store request")

        filters = draft.saved_filters or {}
        ecastings_list = filters.get('requested_ecastings', [])
        if any(t.get('talent_id') == talent.talent_id for t in ecastings_list):
            raise HTTPException(status_code=400, detail="E-casting already added")

        talent_snapshot = {
            "talent_id": talent.talent_id,
            "job_id": None,
            "name": talent.name,
            "role": talent.role,
            "gender": talent.gender,
            "location": talent.location,
            "country": talent.country,
            "continent": talent.continent,
            "is_active": talent.is_active,
            "approval_status": talent.approval_status,
            "is_available": talent.is_available,
            "is_available_on_request": talent.is_available_on_request,
            "agent_id": talent.agent_id,
            "agent_name": talent.agent.full_name if talent.agent else "Unknown",
            "images": [f"/media/{img.image}" for img in sorted(talent.images, key=lambda x: x.image_id)] if talent.images else [],
            "eye_color": talent.eye_colour,
            "hair_type": talent.hair_type,
            "hair_color": talent.hair_colour,
            "skin_color": talent.skin_color,
            "height": talent.height, "bust": talent.bust, "waist": talent.waist, "hips": talent.hips,
            "shoe_size": talent.shoe_size, "dress_size": talent.dress_size,
            "available_dates": [ad.available_date for ad in talent.available_dates if ad.is_active]
        }
        filters['requested_ecastings'] = ecastings_list + [talent_snapshot]
        filters['last_updated_timestamp'] = datetime.now(timezone.utc).isoformat()
        draft.saved_filters = filters
        flag_modified(draft, "saved_filters")
        db.commit()
        return {
            "status_code": 200, 
            "status_message": f"E-casting requested for {talent.name} (Pending job generation)"
        }

    db.refresh(job)

    ai_result = db.query(JobAIResult).filter(JobAIResult.job_id == job.job_id).first()
    if not ai_result:
        ai_result = JobAIResult(job_id=job.job_id, requested_ecastings=[])
        db.add(ai_result)
    
    ecastings_list = ai_result.requested_ecastings or []
    
    if any(t.get('talent_id') == talent.talent_id for t in ecastings_list):
        raise HTTPException(status_code=400, detail="E-casting already added")

    job.ecastings_count = (job.ecastings_count or 0) + 1

    talent_snapshot = {
        "talent_id": talent.talent_id,
        "job_id": job.job_id,
        "name": talent.name,
        "role": talent.role,
        "gender": talent.gender,
        "location": talent.location,
        "country": talent.country,
        "continent": talent.continent,
        "is_active": talent.is_active,
        "approval_status": talent.approval_status,
        "is_available": talent.is_available,
        "is_available_on_request": talent.is_available_on_request,
        "agent_id": talent.agent_id,
        "agent_name": talent.agent.full_name if talent.agent else "Unknown",
        "images": [f"/media/{img.image}" for img in sorted(talent.images, key=lambda x: x.image_id)] if talent.images else [],
        "eye_color": talent.eye_colour,
        "hair_type": talent.hair_type,
        "hair_color": talent.hair_colour,
        "skin_color": talent.skin_color,
        "height": talent.height, "bust": talent.bust, "waist": talent.waist, "hips": talent.hips,
        "shoe_size": talent.shoe_size, "dress_size": talent.dress_size,
        "available_dates": [ad.available_date for ad in talent.available_dates if ad.is_active]
    }
    
    ai_result.requested_ecastings = list(ecastings_list) + [talent_snapshot]

    # Update session draft timestamp
    if job.session and job.session.draft:
        job.session.draft.saved_filters['last_updated_timestamp'] = datetime.now(timezone.utc).isoformat()
        flag_modified(job.session.draft, "saved_filters")

    db.commit()
    return {
        "status_code": 200, 
        "status_message": f"E-casting requested for {talent.name}"
    }

#########--------request for polas--------##########

@app.post("/api/jobs/request-polas", dependencies=[Depends(limiter)])
async def request_polas(
    request: RequestTalentJobRequest,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Endpoint to request polas for a specific talent in a job."""
    job = None
    if request.job_id:
        job = db.query(Job).filter(Job.job_id == request.job_id, Job.job_created_by_id == user_id).first()
    elif request.session_id:
        job = db.query(Job).filter(Job.session_id == request.session_id, Job.job_created_by_id == user_id).order_by(Job.created_at.desc()).first()

    talent = db.query(Talent).filter(Talent.talent_id == request.talent_id).first()
    if not talent:
        raise HTTPException(status_code=404, detail="Talent not found")

    if not job:
        if not request.session_id:
            raise HTTPException(status_code=400, detail="Job not found and no session_id provided")
        draft = db.query(Draft).filter(Draft.session_id == request.session_id, Draft.user_id == user_id).first()
        if not draft:
            raise HTTPException(status_code=404, detail="Session not found to store request")

        filters = draft.saved_filters or {}
        polas_list = filters.get('requested_polas', [])
        if any(t.get('talent_id') == talent.talent_id for t in polas_list):
            raise HTTPException(status_code=400, detail="Polas already added")

        talent_snapshot = {
            "talent_id": talent.talent_id,
            "job_id": None,
            "name": talent.name,
            "role": talent.role,
            "gender": talent.gender,
            "location": talent.location,
            "country": talent.country,
            "continent": talent.continent,
            "is_active": talent.is_active,
            "approval_status": talent.approval_status,
            "is_available": talent.is_available,
            "is_available_on_request": talent.is_available_on_request,
            "agent_id": talent.agent_id,
            "agent_name": talent.agent.full_name if talent.agent else "Unknown",
            "images": [f"/media/{img.image}" for img in sorted(talent.images, key=lambda x: x.image_id)] if talent.images else [],
            "eye_color": talent.eye_colour,
            "hair_type": talent.hair_type,
            "hair_color": talent.hair_colour,
            "skin_color": talent.skin_color,
            "height": talent.height, "bust": talent.bust, "waist": talent.waist, "hips": talent.hips,
            "shoe_size": talent.shoe_size, "dress_size": talent.dress_size,
            "available_dates": [ad.available_date for ad in talent.available_dates if ad.is_active]
        }
        filters['requested_polas'] = polas_list + [talent_snapshot]
        filters['last_updated_timestamp'] = datetime.now(timezone.utc).isoformat()
        draft.saved_filters = filters
        flag_modified(draft, "saved_filters")
        db.commit()
        return {
            "status_code": 200, 
            "status_message": f"Polas requested for {talent.name} (Pending job generation)"
        }

    db.refresh(job)

    ai_result = db.query(JobAIResult).filter(JobAIResult.job_id == job.job_id).first()
    if not ai_result:
        ai_result = JobAIResult(job_id=job.job_id, requested_polas=[])
        db.add(ai_result)
    
    polas_list = ai_result.requested_polas or []
    
    if any(t.get('talent_id') == talent.talent_id for t in polas_list):
        raise HTTPException(status_code=400, detail="Polas already added")

    job.polas_count = (job.polas_count or 0) + 1

    talent_snapshot = {
        "talent_id": talent.talent_id,
        "job_id": job.job_id,
        "name": talent.name,
        "role": talent.role,
        "gender": talent.gender,
        "location": talent.location,
        "country": talent.country,
        "continent": talent.continent,
        "is_active": talent.is_active,
        "approval_status": talent.approval_status,
        "is_available": talent.is_available,
        "is_available_on_request": talent.is_available_on_request,
        "agent_id": talent.agent_id,
        "agent_name": talent.agent.full_name if talent.agent else "Unknown",
        "images": [f"/media/{img.image}" for img in sorted(talent.images, key=lambda x: x.image_id)] if talent.images else [],
        "eye_color": talent.eye_colour,
        "hair_type": talent.hair_type,
        "hair_color": talent.hair_colour,
        "skin_color": talent.skin_color,
        "height": talent.height, "bust": talent.bust, "waist": talent.waist, "hips": talent.hips,
        "shoe_size": talent.shoe_size, "dress_size": talent.dress_size,
        "available_dates": [ad.available_date for ad in talent.available_dates if ad.is_active]
    }
    
    ai_result.requested_polas = list(polas_list) + [talent_snapshot]

    # Update session draft timestamp
    if job.session and job.session.draft:
        job.session.draft.saved_filters['last_updated_timestamp'] = datetime.now(timezone.utc).isoformat()
        flag_modified(job.session.draft, "saved_filters")

    db.commit()
    return {
        "status_code": 200, 
        "status_message": f"Polas requested for {talent.name}"
    }

@app.post("/api/talents/shortlist", dependencies=[Depends(limiter)])
async def shortlist_talent(
    request: ShortlistTalentRequest,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Shortlist a talent for a specific job."""
    talent = db.query(Talent).filter(Talent.talent_id == request.talent_id).first()
    if not talent:
        raise HTTPException(status_code=404, detail="Talent not found")

    job = None
    if request.job_id:
        job = db.query(Job).filter(Job.job_id == request.job_id, Job.job_created_by_id == user_id).first()
    elif request.session_id:
        job = db.query(Job).filter(Job.session_id == request.session_id, Job.job_created_by_id == user_id).order_by(Job.created_at.desc()).first()

    shortlist_query = db.query(ShortlistedTalent).filter(
        ShortlistedTalent.user_id == user_id,
        ShortlistedTalent.talent_id == request.talent_id
    )
    if job:
        shortlist_query = shortlist_query.filter(ShortlistedTalent.job_id == job.job_id)
    elif request.session_id:
        shortlist_query = shortlist_query.filter(ShortlistedTalent.session_id == request.session_id, ShortlistedTalent.job_id == None)
    else:
        raise HTTPException(status_code=400, detail="Missing job_id or session_id") 

    existing_shortlist = shortlist_query.first()

    if existing_shortlist:
        raise HTTPException(status_code=400, detail=f"Talent {talent.name} already shortlisted.")

    if job:
        job.shortlisted_count = (job.shortlisted_count or 0) + 1 

    new_shortlist = ShortlistedTalent(
        session_id=request.session_id,
        user_id=user_id,
        talent_id=request.talent_id,
        job_id=job.job_id if job else None
    )
    db.add(new_shortlist)
    db.commit()
    return {
        "status_code": 200, 
        "status_message": f"Talent {talent.name} shortlisted."
    }

# ########--------View calendar of a member that shows which dates they are available---------#########

# @app.get("/talent/{talent_id}/availability")
# async def get_availability(
#     talent_id: int, 
#     db: Session = Depends(get_db)
#     ):
#     """
#     View availability dates of a member
#     """
#     talent = db.query(Talent).filter(Talent.talent_id == talent_id).first()
#     if not talent: raise HTTPException(404)
#     return {
#         "talent_id": talent.talent_id,
#         "name": talent.name,
#         "is_active": talent.is_active,
#         "available_on": talent.available_date
#     }

    
@app.post("/api/talents/book", dependencies=[Depends(limiter)])
async def book_talent(
    request: BookTalentRequest,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Book a talent. Saves the talent if not already saved."""
    talent = db.query(Talent).filter(Talent.talent_id == request.talent_id).first()
    if not talent:
        raise HTTPException(status_code=404, detail="Talent not found")

    job = None
    if request.job_id:
        job = db.query(Job).filter(Job.job_id == request.job_id, Job.job_created_by_id == user_id).first()
    elif request.session_id:
        job = db.query(Job).filter(Job.session_id == request.session_id, Job.job_created_by_id == user_id).order_by(Job.created_at.desc()).first()

    # Check if already booked
    booking_query = db.query(Booking).filter(
        Booking.user_id == user_id,
        Booking.talent_id == request.talent_id
    )
    if job:
        booking_query = booking_query.filter(Booking.job_id == job.job_id)
    elif request.session_id:
        booking_query = booking_query.filter(Booking.session_id == request.session_id, Booking.job_id == None)
    else:
        raise HTTPException(status_code=400, detail="Missing job_id or session_id")

    existing_booking = booking_query.first()
    if existing_booking:
        raise HTTPException(status_code=400, detail=f"Talent {talent.name} already booked.")

    # Get booking date
    booking_date = date.today()

    shortlist_query = db.query(ShortlistedTalent).filter(
        ShortlistedTalent.user_id == user_id,
        ShortlistedTalent.talent_id == request.talent_id
    )
    if job:
        shortlist_query = shortlist_query.filter(ShortlistedTalent.job_id == job.job_id)
    else:
        shortlist_query = shortlist_query.filter(ShortlistedTalent.session_id == request.session_id, ShortlistedTalent.job_id == None)

    existing_shortlist = shortlist_query.first()

    if not existing_shortlist:
        db.add(ShortlistedTalent(
            user_id=user_id, 
            talent_id=request.talent_id, 
            session_id=request.session_id,
            job_id=job.job_id if job else None
        ))
        if job:
            job.shortlisted_count = (job.shortlisted_count or 0) + 1

    new_booking = Booking(
        session_id=request.session_id,
        user_id=user_id,
        talent_id=request.talent_id,
        job_id=job.job_id if job else None
    )
    db.add(new_booking)
    db.commit()

    # Set is_active to false for the booking date
    booking_date = new_booking.created_at.date()
    db.query(TalentAvailableDate).filter(
        TalentAvailableDate.talent_id == request.talent_id,
        TalentAvailableDate.available_date == booking_date
    ).update({"is_active": False})
    db.commit()

    return {
        "status_code": 200, 
        "status_message": f"Talent {talent.name} booked successfully."
    }


@app.get("/api/view-shortlists", dependencies=[Depends(limiter)])
async def get_shortlist(
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Returns a list of jobs with a preview of shortlisted talents.
    """
    total_jobs = db.query(Job).filter(Job.job_created_by_id == user_id).count()
    queries = db.query(Job).filter(Job.job_created_by_id == user_id, Job.shortlisted_count > 0)
    
    shortlists_data = []
    now = datetime.now(timezone.utc)
    
    for job in queries:
        shortlisted_records = db.query(ShortlistedTalent)\
            .filter(ShortlistedTalent.job_id == job.job_id, ShortlistedTalent.user_id == user_id)\
            .order_by(ShortlistedTalent.created_at.desc())\
            .limit(5).all()
        
        preview_talents = []
        for record in shortlisted_records:
            talent = record.talent
            img_url = None
            if talent.images:
                first_img = sorted(talent.images, key=lambda x: x.image_id)[0]
                img_url = f"/media/{first_img.image}"
            
            preview_talents.append(TalentPreview(
                talent_id=str(talent.talent_id),
                profile_image_url=img_url,
                is_active=talent.is_active,
                approval_status=talent.approval_status,
                is_available=talent.is_available,
                is_available_on_request=talent.is_available_on_request
            ))
        
        elapsed_hours = (now - job.created_at).total_seconds() / 3600
        remaining_hours = max(0, int(72 - elapsed_hours))
        
        shortlists_data.append(ShortlistSummaryItem(
            job_id=str(job.job_id),
            job_title=job.title,
            talent_count=job.shortlisted_count,
            time_remaining_hours=remaining_hours,
            preview_talents=preview_talents,
            extra_talent_count=max(0, job.shortlisted_count - len(preview_talents))
        ))
        
    return jsonable_encoder(
        {
            "status_code": 200,
            "status_message": "Success",
            **ShortlistSummaryResponse(
                shortlists=shortlists_data,
                pagination=SummaryPagination(
                    page=1, limit=10, total=total_jobs
                )
            ).dict()
        }
    )

@app.post("/api/jobs/selftape/action", dependencies=[Depends(limiter)])
async def update_selftape_status(
    request: SelfTapeStatusAction,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Accept or Reject a self-tape request."""
    if request.status not in ["accepted", "rejected"]:
        raise HTTPException(status_code=400, detail="Invalid status. Use 'accepted' or 'rejected'.")

    if request.job_id:
        job = db.query(Job).filter(Job.job_id == request.job_id, Job.job_created_by_id == user_id).first()
    elif request.session_id:
        job = db.query(Job).filter(Job.session_id == request.session_id, Job.job_created_by_id == user_id).order_by(Job.created_at.desc()).first()
    else:
        raise HTTPException(status_code=400, detail="Missing job_id or session_id")

    talent = db.query(Talent).filter(Talent.talent_id == request.talent_id).first()
    if not talent or not job:
        raise HTTPException(status_code=404, detail="Talent or Job not found.")

    db.refresh(job)

    if user_id != job.job_created_by_id:
        raise HTTPException(status_code=403, detail="Only the job creator can perform this action.")

    ai_result = db.query(JobAIResult).filter(JobAIResult.job_id == job.job_id).first()
    if not ai_result or not ai_result.requested_selftapes:
        raise HTTPException(status_code=404, detail="Self-tape requests not found for this job.")

    selftapes = ai_result.requested_selftapes or []
    talent_snapshot = next((t for t in selftapes if t.get('talent_id') == request.talent_id), None)
    if not talent_snapshot:
        raise HTTPException(status_code=404, detail="Self-tape request not found in job snapshot.")

    st_request = db.query(SelfTapeRequest).filter(
        SelfTapeRequest.job_id == job.job_id,
        SelfTapeRequest.talent_id == request.talent_id
    ).first()

    if not st_request:
        st_request = SelfTapeRequest(job_id=job.job_id, talent_id=request.talent_id, status="requested")
        db.add(st_request)
        db.flush()

    if st_request.status == request.status:
        raise HTTPException(status_code=400, detail=f"This request is already {request.status}.")

    st_request.status = request.status
    talent_snapshot['status'] = request.status
    flag_modified(ai_result, "requested_selftapes")
    
    # Update session draft timestamp
    if job.session and job.session.draft:
        job.session.draft.saved_filters['last_updated_timestamp'] = datetime.now(timezone.utc).isoformat()
        flag_modified(job.session.draft, "saved_filters")

    db.commit()
    return {
        "status_code": 200, 
        "status_message": f"Self-tape {request.status}."
    }

# @app.get("/api/jobs/selftape/details", response_model=SelfTapeUploadPageResponse, dependencies=[Depends(limiter)])
# async def get_selftape_upload_details(
#     job_id: int,
#     talent_id: int,
#     user_id: int = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Fetch details for the self-tape response/upload page."""
#     talent = db.query(Talent).filter(Talent.talent_id == talent_id).first()
#     job = db.query(Job).filter(Job.job_id == job_id).first()
#     if not talent or not job:
#         raise HTTPException(status_code=404, detail="Talent or Job not found.")

#     if user_id not in [talent.talent_id, talent.agent_id, job.job_created_by_id]:
#         raise HTTPException(status_code=403, detail="Unauthorized to access these details.")

#     st_request = db.query(SelfTapeRequest).options(joinedload(SelfTapeRequest.tapes)).filter(
#         SelfTapeRequest.job_id == job_id,
#         SelfTapeRequest.talent_id == talent_id
#     ).first()

#     ai_result = db.query(JobAIResult).filter(JobAIResult.job_id == job_id).first()
#     if not st_request or not ai_result:
#         raise HTTPException(status_code=404, detail="Job or AI results not found.")

#     return SelfTapeUploadPageResponse(
#         talent_name=talent.name,
#         talent_role=talent.role,
#         job_title=job.title,
#         job_budget=job.budget,
#         timeline=ai_result.shoot_date,
#         status=st_request.status,
#         existing_tapes=[t.tape_url for t in st_request.tapes]
#     )

@app.post("/api/jobs/selftape/upload", dependencies=[Depends(limiter)])
async def upload_selftape_videos(
    job_id: Optional[int] = Form(None),
    session_id: Optional[str] = Form(None),
    talent_id: int = Form(...),
    files: List[UploadFile] = File([]),
    tape_urls: List[str] = Form([]),
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload tapes for a self-tape and mark status as 'responded'."""
    if job_id:
        job = db.query(Job).filter(Job.job_id == job_id).first()
    elif session_id:
        job = db.query(Job).filter(Job.session_id == session_id, Job.job_created_by_id == user_id).order_by(Job.created_at.desc()).first()
    else:
        raise HTTPException(status_code=400, detail="Missing job_id or session_id")

    talent = db.query(Talent).filter(Talent.talent_id == talent_id).first()
    if not talent or not job:
        raise HTTPException(status_code=404, detail="Talent or Job not found.")

    if user_id not in [talent.talent_id, talent.agent_id, job.job_created_by_id]:
        raise HTTPException(status_code=403, detail="Unauthorized to upload tapes for this talent.")

    ai_result = db.query(JobAIResult).filter(JobAIResult.job_id == job.job_id).first()
    if not ai_result:
        raise HTTPException(status_code=404, detail="Job AI results not found.")

    selftapes = ai_result.requested_selftapes or []
    talent_snapshot = next((t for t in selftapes if t.get('talent_id') == talent_id), None)
    if not talent_snapshot:
        raise HTTPException(status_code=404, detail="Self-tape request not found in job snapshot.")

    st_request = db.query(SelfTapeRequest).options(joinedload(SelfTapeRequest.tapes)).filter(
        SelfTapeRequest.job_id == job.job_id,
        SelfTapeRequest.talent_id == talent_id
    ).first()

    if not st_request:
        st_request = SelfTapeRequest(job_id=job.job_id, talent_id=talent_id, status="requested")
        db.add(st_request)
        db.flush()

    if st_request.status == "responded":
        raise HTTPException(status_code=400, detail="You have already responded to this request.")

    st_request.status = 'responded'
    talent_snapshot['status'] = 'responded'

    upload_dir = "media/selftapes"
    os.makedirs(upload_dir, exist_ok=True)
    
    uploaded_file_urls = []
    if files:
        for file in files:
            allowed_video_exts = [".mp4", ".mov", ".avi", ".wmv", ".m4v"]
            file_ext = os.path.splitext(file.filename)[1].lower()

            if not file.content_type.startswith("video/") or file_ext not in allowed_video_exts:
                raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a valid video format. Supported: {', '.join(allowed_video_exts)}")

            unique_filename = f"{uuid.uuid4()}{file_ext}"
            file_path = os.path.join(upload_dir, unique_filename)
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded_file_urls.append(f"/media/selftapes/{unique_filename}")

    processed_external_urls = []
    if tape_urls:
        for item in tape_urls:
            if not item: 
                continue
       
            if isinstance(item, str) and item.strip().startswith("[") and item.strip().endswith("]"):
                try:
                    parsed = json.loads(item.strip())
                    if isinstance(parsed, list):
                        processed_external_urls.extend([str(x) for x in parsed if x])
                        continue 
                except (json.JSONDecodeError, TypeError):
                    pass # Not a valid JSON list, proceed to next check

            if isinstance(item, str) and "," in item:
                split_urls = [url.strip() for url in item.split(',') if url.strip()]
                processed_external_urls.extend(split_urls)
            else:
                processed_external_urls.append(item.strip())

    existing_json_tapes = list(talent_snapshot.get('tapes') or [])
    
    all_new_urls = uploaded_file_urls + processed_external_urls
    
    if not all_new_urls:
        raise HTTPException(status_code=400, detail="No files or video URLs provided.")

    for url in all_new_urls:
        if not any(link.tape_url == url for link in st_request.tapes):
            new_link = SelfTapeLink(tape_url=url)
            st_request.tapes.append(new_link)
        
        if url not in existing_json_tapes:
            existing_json_tapes.append(url)
    
    talent_snapshot['tapes'] = existing_json_tapes
    ai_result.requested_selftapes = selftapes  
    flag_modified(ai_result, "requested_selftapes")

    # Update session draft timestamp
    if job.session and job.session.draft:
        job.session.draft.saved_filters['last_updated_timestamp'] = datetime.now(timezone.utc).isoformat()
        flag_modified(job.session.draft, "saved_filters")

    db.commit()
    return {
        "status_code": 200,
        "status_message": "Self-tapes uploaded and status updated to responded.",
        "uploaded_urls": all_new_urls
    }

@app.post("/api/jobs/polas/action", dependencies=[Depends(limiter)])
async def update_pola_status(
    request: PolaStatusAction,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Accept or Reject a polas request."""
    if request.status not in ["accepted", "rejected"]:
        raise HTTPException(status_code=400, detail="Invalid status. Use 'accepted' or 'rejected'.")

    if request.job_id:
        job = db.query(Job).filter(Job.job_id == request.job_id, Job.job_created_by_id == user_id).first()
    elif request.session_id:
        job = db.query(Job).filter(Job.session_id == request.session_id, Job.job_created_by_id == user_id).order_by(Job.created_at.desc()).first()
    else:
        raise HTTPException(status_code=400, detail="Missing job_id or session_id")

    talent = db.query(Talent).filter(Talent.talent_id == request.talent_id).first()
    if not talent or not job:
        raise HTTPException(status_code=404, detail="Talent or Job not found.")

    db.refresh(job)

    if user_id != job.job_created_by_id:
        raise HTTPException(status_code=403, detail="Only the job creator can perform this action.")

    ai_result = db.query(JobAIResult).filter(JobAIResult.job_id == job.job_id).first()
    if not ai_result or not ai_result.requested_polas:
        raise HTTPException(status_code=404, detail="Pola requests not found for this job.")

    polas = ai_result.requested_polas or []
    talent_snapshot = next((t for t in polas if t.get('talent_id') == request.talent_id), None)
    if not talent_snapshot:
        raise HTTPException(status_code=404, detail="Polas request not found in job snapshot.")

    st_request = db.query(PolaRequest).filter(
        PolaRequest.job_id == job.job_id,
        PolaRequest.talent_id == request.talent_id
    ).first()

    if not st_request:
        st_request = PolaRequest(job_id=job.job_id, talent_id=request.talent_id, status="requested")
        db.add(st_request)
        db.flush()

    st_request.status = request.status
    talent_snapshot['status'] = request.status
    flag_modified(ai_result, "requested_polas")
    
    # Update session draft timestamp
    if job.session and job.session.draft:
        job.session.draft.saved_filters['last_updated_timestamp'] = datetime.now(timezone.utc).isoformat()
        flag_modified(job.session.draft, "saved_filters")

    db.commit()
    return {
        "status_code": 200, 
        "status_message": f"Polas {request.status}."
    }

@app.post("/api/jobs/polas/upload", dependencies=[Depends(limiter)])
async def upload_pola_images(
    job_id: Optional[int] = Form(None),
    session_id: Optional[str] = Form(None),
    talent_id: int = Form(...),
    files: List[UploadFile] = File([]),
    image_urls: List[str] = Form([]),
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload images for polas and mark status as 'responded'."""
    if job_id:
        job = db.query(Job).filter(Job.job_id == job_id).first()
    elif session_id:
        job = db.query(Job).filter(Job.session_id == session_id, Job.job_created_by_id == user_id).order_by(Job.created_at.desc()).first()
    else:
        raise HTTPException(status_code=400, detail="Missing job_id or session_id")

    talent = db.query(Talent).filter(Talent.talent_id == talent_id).first()
    if not talent or not job:
        raise HTTPException(status_code=404, detail="Talent or Job not found.")

    if user_id not in [talent.talent_id, talent.agent_id, job.job_created_by_id]:
        raise HTTPException(status_code=403, detail="Unauthorized to upload polas for this talent.")

    ai_result = db.query(JobAIResult).filter(JobAIResult.job_id == job.job_id).first()
    if not ai_result:
        raise HTTPException(status_code=404, detail="Job AI results not found.")

    polas = ai_result.requested_polas or []
    talent_snapshot = next((t for t in polas if t.get('talent_id') == talent_id), None)
    if not talent_snapshot:
        raise HTTPException(status_code=404, detail="Pola request not found in job snapshot.")

    st_request = db.query(PolaRequest).options(joinedload(PolaRequest.images)).filter(
        PolaRequest.job_id == job.job_id,
        PolaRequest.talent_id == talent_id
    ).first()

    if not st_request:
        st_request = PolaRequest(job_id=job.job_id, talent_id=talent_id, status="requested")
        db.add(st_request)
        db.flush()

    st_request.status = 'responded'
    talent_snapshot['status'] = 'responded'

    upload_dir = "media/polas"
    os.makedirs(upload_dir, exist_ok=True)
    
    uploaded_file_urls = []
    if files:
        for file in files:
            allowed_image_exts = [".jpg", ".jpeg", ".png", ".webp", ".heic"]
            file_ext = os.path.splitext(file.filename)[1].lower()

            if not file.content_type.startswith("image/") or file_ext not in allowed_image_exts:
                raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a valid image format. Supported: {', '.join(allowed_image_exts)}")

            unique_filename = f"{uuid.uuid4()}{file_ext}"
            file_path = os.path.join(upload_dir, unique_filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_file_urls.append(f"/media/polas/{unique_filename}")

    processed_external_urls = []
    if image_urls:
        for item in image_urls:
            if not item: continue
            processed_external_urls.append(item.strip())

    existing_json_tapes = list(talent_snapshot.get('polas') or [])
    all_new_urls = uploaded_file_urls + processed_external_urls
    
    if not all_new_urls:
        raise HTTPException(status_code=400, detail="No files or image URLs provided.")

    for url in all_new_urls:
        if not any(link.pola_url == url for link in st_request.images):
            new_link = PolaLink(pola_url=url)
            st_request.images.append(new_link)
        if url not in existing_json_tapes:
            existing_json_tapes.append(url)
    
    talent_snapshot['polas'] = existing_json_tapes
    ai_result.requested_polas = polas  
    flag_modified(ai_result, "requested_polas")

    # Update session draft timestamp
    if job.session and job.session.draft:
        job.session.draft.saved_filters['last_updated_timestamp'] = datetime.now(timezone.utc).isoformat()
        flag_modified(job.session.draft, "saved_filters")

    db.commit()
    return {
        "status_code": 200,
        "status_message": "Polas uploaded and status updated to responded.",
        "uploaded_urls": all_new_urls
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8008, reload= True)
