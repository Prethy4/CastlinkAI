import os
from dotenv import load_dotenv
load_dotenv(override=True)

from datetime import date, datetime, timezone, timedelta
from fastapi import FastAPI, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage, AIMessage
from database import init_db, get_db, Talent, ChatSession, ChatMessage, Draft, Job, SavedTalent, Booking, JobAIResult
from schemas import TalentResponse, ChatSessionResponse, DraftResponse, ChatRequest, UserRequest, SessionRequest, DraftRequest, JobResponse, GenerateCastingRequest, SaveTalentRequest, BookTalentRequest, ContinueDraftResponse, ChatMessageResponse, JobResultResponse, WrappedChatResponse, ConversationResponse, PaginationResponse, TalentDataResponse, UserDraftResponse, DraftsSavedFilters
from services import app_graph, extract_information, generate_ask_response, CustomEncoder, RateLimiter, time_ago, parse_budget
from typing import List, Optional
import json

limiter = RateLimiter(limit=20, window=60, error_msg="Something went wrong. Please try again later or contact the support.")

MANDATORY_FIELDS = ["location", "shoot_date", "budget", "job_type", "gender", "skin_color"]

# Initialize database
init_db()

# api_key = os.getenv("OPENAI_API_KEY")
# if api_key:
#     print(f"DEBUG: OpenAI API Key loaded. Ends in: ...{api_key[-4:]}")
# else:
#     print("DEBUG: OPENAI_API_KEY not found in environment variables.")

app = FastAPI(title="AI-Powered Casting")

##########-------health check-------##########

@app.get("/health")
async def health_check():
    return {"message" : "server running"}

SERVER_START_TIME = datetime.now(timezone.utc)

@app.post("/api/chat", response_model=WrappedChatResponse, dependencies=[Depends(limiter)])
async def send_message(
    request: ChatRequest,
    db: Session = Depends(get_db)
    ):
    """Chat services for conversation with the chatbot"""
    try:
        message = request.message
        user_id = request.user_id
        session_id = request.session_id
        location = request.location
        shoot_date = request.shoot_date
        budget = request.budget
        job_type = request.job_type
        save_as_draft = request.save_as_draft

        chat_session = None
        if session_id:
            chat_session = db.query(ChatSession).filter(ChatSession.session_id == session_id, ChatSession.user_id == user_id).first()

        if not chat_session:
            chat_session = db.query(ChatSession).filter(ChatSession.user_id == user_id).order_by(ChatSession.created_at.desc()).first()
            if chat_session and chat_session.created_at < SERVER_START_TIME:
                chat_session = None
        
        if not chat_session:
            chat_session = ChatSession(user_id=user_id)
            db.add(chat_session)
            db.commit()
            db.refresh(chat_session)

        user_msg = ChatMessage(session_id=chat_session.session_id, sender="user", content=message)
        db.add(user_msg)
        db.commit()

        past_messages = db.query(ChatMessage).filter(ChatMessage.session_id == chat_session.session_id).order_by(ChatMessage.message_id.desc()).limit(10).all()
        past_messages.reverse()
        msgs = []
        for m in past_messages:
            if m.sender == "user":
                msgs.append(HumanMessage(content=m.content))
            else:
                msgs.append(AIMessage(content=m.content))

        filters = {}
        existing_draft = db.query(Draft).filter(Draft.session_id == chat_session.session_id).first()
        if existing_draft:
            if existing_draft.saved_filters: filters.update(existing_draft.saved_filters)

        if location: filters['location'] = location
        if budget: filters['budget'] = budget
        if job_type: filters['job_type'] = job_type
        if shoot_date:
            cleaned_dates = []
            for d in shoot_date:
                if "," in d:
                    cleaned_dates.extend([dt.strip() for dt in d.split(",") if dt.strip()])
                else:
                    if d.strip():
                        cleaned_dates.append(d)
            if cleaned_dates:
                filters['shoot_date'] = cleaned_dates

        # Data sent with the message is saved to Draft immediately
        draft = db.query(Draft).filter(Draft.session_id == chat_session.session_id).first()
        if not draft:
            draft = Draft(session_id=chat_session.session_id, user_id=user_id)
            db.add(draft)
        
        draft.saved_filters = filters
        draft.location = filters.get("location")
        draft.budget = filters.get("budget")
        draft.job_type = filters.get("job_type")
        s_dates = filters.get("shoot_date")
        if isinstance(s_dates, list):
            draft.shoot_date = ", ".join(s_dates)
        else:
            draft.shoot_date = s_dates
        db.commit()

        extracted_updates = extract_information(message, filters)
        filters.update(extracted_updates)

        # --- Update ChatMessage with specific fields ---
   
        user_msg.location = filters.get("location")
        user_msg.budget = filters.get("budget")
        user_msg.job_type = filters.get("job_type")
        
        s_dates_msg = filters.get("shoot_date")
        if isinstance(s_dates_msg, list):
            user_msg.shoot_date = ", ".join(s_dates_msg)
        else:
            user_msg.shoot_date = s_dates_msg
        db.commit()

        # --- Save as Draft Flow ---
        if save_as_draft:
            
            draft.phase = "draft"
            draft.saved_filters = filters
            
            draft.location = filters.get("location")
            draft.budget = filters.get("budget")
            draft.job_type = filters.get("job_type")
            s_dates = filters.get("shoot_date")
            if isinstance(s_dates, list):
                draft.shoot_date = ", ".join(s_dates)
            else:
                draft.shoot_date = s_dates
            db.commit()
            raise HTTPException(status_code=200, detail="Draft saved successfully.")

        # Check for mandatory fields
        missing_keys = [k for k in MANDATORY_FIELDS if not filters.get(k)]
        
        final_state = {}
        response_content = ""

        if missing_keys:
            response_content = generate_ask_response(missing_keys)
            final_state = {"filters": filters, "messages": [AIMessage(content=response_content)]}
        else:
            inputs = {"messages": msgs, "filters": filters}
            final_state = app_graph.invoke(inputs, config={"recursion_limit": 5})
            last_msg = final_state['messages'][-1]
            response_content = last_msg.content
        
        # --- Persist Session State (Draft) ---
        current_filters = final_state.get('filters', {})
        
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

        current_phase = "READY_TO_GENERATE" if all(k in current_filters for k in MANDATORY_FIELDS) else "COLLECT_MANDATORY"

        draft.phase = current_phase
        draft.saved_filters = json.loads(json.dumps(current_filters, cls=CustomEncoder))

        draft.location = current_filters.get("location")
        draft.budget = current_filters.get("budget")
        draft.job_type = current_filters.get("job_type")
        s_dates = current_filters.get("shoot_date")
        if isinstance(s_dates, list):
            draft.shoot_date = ", ".join(s_dates)
        else:
            draft.shoot_date = s_dates
        db.commit()

        ai_msg = ChatMessage(session_id=chat_session.session_id, sender="ai", content=response_content)
        db.add(ai_msg)
        db.commit()

        final_talents = []
        total_results = 0
        for msg in final_state['messages']:
            if hasattr(msg, 'name') and msg.name == 'generate_casting':
                if hasattr(msg, 'artifact') and msg.artifact:
                    talent_data = msg.artifact.get('talents', [])
                    for t in talent_data:
                        final_talents.append(TalentResponse(**t))
                    total_results = msg.artifact.get('total_results', len(final_talents))
                    break

        page = 1
        per_page = len(final_talents) if final_talents else 10
        has_next = total_results > per_page

        return WrappedChatResponse(
            session_id=chat_session.session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            conversation=ConversationResponse(text=response_content),
            pagination=PaginationResponse(
                total_results=total_results,
                page=page,
                per_page=per_page,
                has_next=has_next
            ),
            data=TalentDataResponse(talents=final_talents)
        )
    except Exception as e:
        db.rollback()
        raise e
        # raise HTTPException(status_code=500, detail="Something went wrong. Please try again later or contact the support.")

###########----------chat session-----------############

@app.get("/api/chat/sessions", response_model=List[ChatSessionResponse], dependencies=[Depends(limiter)])
async def get_sessions(
    request: UserRequest = Depends(), 
    db: Session = Depends(get_db)
    ):
    
    sessions = db.query(ChatSession).filter(ChatSession.user_id == request.user_id).all()
    if not sessions:
        raise HTTPException(status_code=404, detail="User not found")
    return sessions

############----------get chat session by id--------############ recheck

@app.get("/api/chat/session-id", response_model=ChatSessionResponse, dependencies=[Depends(limiter)])
async def get_session_details(
    request: SessionRequest = Depends(), 
    db: Session = Depends(get_db)
    ):
    
    session = db.query(ChatSession).filter(ChatSession.session_id == request.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat not found")
    return session

###########----------delete chat-----------############

@app.delete("/api/chat/delete-session-id", dependencies=[Depends(limiter)])
async def delete_session(
    request: SessionRequest, 
    db: Session = Depends(get_db)
    ):

        session = db.query(ChatSession).filter(ChatSession.session_id == request.session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Chat not found")
        db.delete(session)
        db.commit()
        return {"status": "success", "message": "Chat deleted"}
    
#############----------retrives all draft state------------###############

@app.get("/api/chat/drafts", response_model=List[UserDraftResponse], dependencies=[Depends(limiter)])
async def get_user_drafts(
    user_id: int,
    search: Optional[str] = Query(None, alias="search", description="Search drafts by job type"),
    db: Session = Depends(get_db)
    ):
        query = db.query(Draft).filter(Draft.user_id == user_id, Draft.phase != "generated")
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

        if not drafts:
            raise HTTPException(status_code=404, detail="No drafts found")
        
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

            saved_filters_response = DraftsSavedFilters(
                job_type=job_type,
                message=message_content
            )

            if not draft.last_updated:
                draft.last_updated = str(datetime.now())

            response_drafts.append(
                UserDraftResponse(
                    draft_id=draft.draft_id,
                    user_id=draft.user_id,
                    session_id=draft.session_id,
                    saved_filters=saved_filters_response,
                    Updated=time_ago(draft.last_updated),
                    last_updated=draft.last_updated
                )
            )
                
        return response_drafts

#############----------retrives draft state------------###############

@app.get("/api/chat/draft-id", response_model=DraftResponse, dependencies=[Depends(limiter)])
async def get_draft(
    request: DraftRequest = Depends(), 
    db: Session = Depends(get_db)
    ):
    
    draft = db.query(Draft).filter(Draft.draft_id == request.draft_id, Draft.user_id == request.user_id).first()
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    
    messages_until_draft = db.query(ChatMessage).filter(
        ChatMessage.session_id == draft.session_id,
        ChatMessage.timestamp <= (draft.last_updated or datetime.now(timezone.utc))
    ).order_by(ChatMessage.message_id.asc()).all()
        
    return DraftResponse(
        draft_id=draft.draft_id,
        user_id=draft.user_id,
        session_id=draft.session_id,
        saved_filters=draft.saved_filters or {},
        last_updated=draft.last_updated,
        messages=[ChatMessageResponse.from_orm(m) for m in messages_until_draft]
    )

############--------continue draft------##############

@app.get("/api/chat/continue-draft-id", response_model=ContinueDraftResponse, dependencies=[Depends(limiter)])
async def continue_draft(
    request: DraftRequest = Depends(),
    db: Session = Depends(get_db)
    ):
    
    draft = db.query(Draft).filter(Draft.draft_id == request.draft_id, Draft.user_id == request.user_id).first()
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    
    session = db.query(ChatSession).filter(ChatSession.session_id == draft.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Associated session not found")
        
    return ContinueDraftResponse(
        session_id=session.session_id,
        messages=[ChatMessageResponse(sender=m.sender, content=m.content, timestamp=m.timestamp) for m in session.messages],
        saved_filters=draft.saved_filters or {}
    )

###########---------delete draft---------############

@app.delete("/api/chat/delete-draft-id", dependencies=[Depends(limiter)])
async def delete_draft(
    request: DraftRequest = Depends(), 
    db: Session = Depends(get_db)
    ):
    
    draft = db.query(Draft).filter(Draft.draft_id == request.draft_id, Draft.user_id == request.user_id).first()
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    
    db.delete(draft)
    db.commit()
    return {"status": "success", "message": "Draft deleted"}

#############----------generate casting job------------###############

@app.post("/api/jobs/generate", response_model=JobResponse, dependencies=[Depends(limiter)])
async def generate_casting_job(
    request: GenerateCastingRequest,
    db: Session = Depends(get_db)
    ):
    """
    Finalize job and create active job record.
    """
    # Retrieve session and draft filters
    draft = db.query(Draft).filter(Draft.session_id == request.session_id).first()
    
    if not draft:
        session = db.query(ChatSession).filter(ChatSession.session_id == request.session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        draft = Draft(session_id=request.session_id, user_id=request.user_id, phase="draft", saved_filters={}, last_updated=str(datetime.now()))
        db.add(draft)
        db.commit()
        db.refresh(draft)
    
    filters = draft.saved_filters or {}
    
    # Fetch from columns (source of truth) or fallback to filters
    d_location = draft.location or filters.get("location")
    d_job_type = draft.job_type or filters.get("job_type")
    
    d_shoot_date = draft.shoot_date
    if not d_shoot_date:
        s_dates = filters.get("shoot_date")
        if isinstance(s_dates, list):
            d_shoot_date = ", ".join(s_dates)
        else:
            d_shoot_date = s_dates

    missing_fields = []
    if not d_location: missing_fields.append("location")
    if not d_job_type: missing_fields.append("job_type")
    if not d_shoot_date: missing_fields.append("shoot_date")

    if missing_fields:
        raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing_fields)}")
    
    suggested_talents_list = filters.get('suggested_talents_list') or []
    total_applicants = len(suggested_talents_list)

    first_msg = db.query(ChatMessage).filter(ChatMessage.session_id == request.session_id).order_by(ChatMessage.message_id.asc()).first()
    first_message_content = first_msg.content if first_msg else None

    budget_min, budget_max = parse_budget(filters.get("budget"))
    
    # Create Job Record
    new_job = Job(
        job_created_by_id=request.user_id,
        session_id=request.session_id,
        title=f"{d_job_type} Casting in {d_location}",
        description=first_message_content or "No description provided",
        location=d_location,
        budget_min=budget_min,
        budget_max=budget_max,
        job_type=d_job_type,
        status=filters.get("status") or "active",
        applicants_count=total_applicants,
        shortlisted_count=0,
        selftapes_count=0
    )
    
    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    # Save AI results in separate table
    ai_result = JobAIResult(
        job_id=new_job.job_id,
        suggested_talents=suggested_talents_list,
        shoot_date=d_shoot_date
    )
    db.add(ai_result)

    draft.phase = "generated"
    db.commit()
    
    return new_job

##########-------------get generated jobs-----------############

@app.get("/api/retrive-generated-jobs", response_model=List[JobResponse], dependencies=[Depends(limiter)])
async def get_user_jobs(
    request: UserRequest = Depends(),
    search: Optional[str] = Query(None, description="Search by job type"),
    sort: Optional[str] = Query("all", description="Sort options: all, urgent, this week"),
    db: Session = Depends(get_db)
    ):
    """
    Return all jobs for a user.
    """
    query = db.query(Job).filter(Job.job_created_by_id == request.user_id)

    if search:
        query = query.filter(Job.job_type.ilike(f"%{search}%"))

    if sort and sort.lower() != 'all':
        now = datetime.now(timezone.utc)
        if sort.lower() == 'urgent':
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            query = query.filter(Job.created_at >= start_of_day)
        elif sort.lower() == 'this week':
            week_ago = now - timedelta(days=7)
            query = query.filter(Job.created_at >= week_ago)

    query = query.order_by(Job.created_at.desc())
    
    return query.all()

# @app.get("/api/jobs/{job_id}", response_model=JobResultResponse, dependencies=[Depends(limiter)])
# async def get_job_result(
#     job_id: int,
#     request: UserRequest = Depends(),
#     db: Session = Depends(get_db)
# ):
#     job = db.query(Job).filter(Job.id == job_id, Job.user_id == request.user_id).first()
#     if not job:
#         raise HTTPException(status_code=404, detail="Job not found")

#     return JobResultResponse(
#         id=job.id,
#         user_id=job.user_id,
#         status=job.status,
#         job_type=job.job_type,
#         first_message=job.first_message,
#         location=job.location,
#         shoot_date=job.shoot_date,
#         budget_range=job.budget_range,
#         total_applicants=job.total_applicants,
#         total_shortlisted=job.shortlisted_count,
#         total_selftapes=job.selftape_count,
#         suggested_talents=[TalentResponse(**t) for t in (job.suggested_talents or [])]
#     )

############-----------view AI results------------###############

@app.get("/api/jobs/view-ai-result", response_model=JobResultResponse, dependencies=[Depends(limiter)])
async def view_ai_result(
    job_id: int,
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    View the final AI result for a specific created job, including the list of suggested talents.
    """
    job = db.query(Job).filter(Job.job_id == job_id, Job.job_created_by_id == user_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    ai_result = db.query(JobAIResult).filter(JobAIResult.job_id == job.job_id).first()
    suggested_talents = ai_result.suggested_talents if ai_result else []
    shoot_date = ai_result.shoot_date if ai_result else None

    messages = []
    if job.session_id:
        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == job.session_id,
            ChatMessage.timestamp <= job.created_at
        ).order_by(ChatMessage.message_id.asc()).all()

    response = JobResultResponse(
        **job.__dict__,
        shoot_date=shoot_date,
        suggested_talents=[TalentResponse(**t) for t in (suggested_talents or [])],
        messages=[ChatMessageResponse.from_orm(m) for m in messages]
    )
    return response

    
##########---------save a talent-------############ 

@app.post("/save-talent")
async def save_talents(
    request: SaveTalentRequest, 
    db: Session = Depends(get_db)
    ):
    """ 
    save a talent for further view
    """
    # Check if talent exists
    talent = db.query(Talent).filter(Talent.talent_id == request.talent_id).first()
    if not talent:
        raise HTTPException(status_code=404, detail="Talent not found")
    
    saved = SavedTalent(
        user_session_id=request.session_id,
        user_id=request.user_id,
        talent_id=request.talent_id,
        saved_at=str(date.today())
    )
    db.add(saved)
    db.commit()
    return {
            "status": "success", 
            "message": f"Talent {talent.name} saved."
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
#         "is_available": talent.is_available,
#         "available_on": talent.available_date
#     }

# # # @app.get("/talent/{talent_id}/calendar")
# # # async def get_calendar(talent_id: str, db: Session = Depends(get_db)):
# # #     """
# # #     View calendar of a member that shows which dates they are available.
# # #     """
# # #     talent = db.query(Talent).filter(Talent.id == talent_id).first()
# # #     if not talent:
# # #         raise HTTPException(status_code=404, detail="Talent not found")
# # #     
# # #     return {
# # #         "talent_id": talent.id,
# # #         "name": talent.name,
# # #         "availability": talent.availability
# # #     }

# #########----------upload selftape endpoint---------############

# # @app.post("/talent/{talent_id}/selftape")
# # async def upload_selftape(
# #     talent_id: str
# #     ):
# #     """
# #     request for an selftape
# #     """
# #     return {
# #             "message": "Self-tape upload endpoint (add file handling here)"
# #             }

# # @app.get("/talent/{talent_id}/request_virtual_casting")
# # async def get_request_virtual_casting(
# #     talent_id: str, 
# #     db: Session = Depends(get_db)
# #     ):
# #     """
# #     request a virtual meet for casting
# #     """
# #     talent = db.query(Talent).filter(Talent.id == talent_id).first()
# #     if not talent: raise HTTPException(404)
# #     return {
# #             "virtual meet endpoint": talent.virtual_meet
# #             }

# # @app.post("/talent/{talent_id}/polas")
# # async def request_polas(
# #     talent_id: str
# #     ):
# #     """
# #     upload polas raw face images
# #     """
# #     return {
# #             "message": "polas request endpoint"
# #             }
    
@app.post("/book-talent")
async def book_talent(
    request: BookTalentRequest, 
    db: Session = Depends(get_db)
    ):
    """
    Book a talent. Automatically saves the talent if not already saved.
    """
    # To check if talent exists
    talent = db.query(Talent).filter(Talent.talent_id == request.talent_id).first()
    if not talent:
        raise HTTPException(status_code=404, detail="Talent not found")

    # To check if already saved
    saved = db.query(SavedTalent).filter(
        SavedTalent.user_id == request.user_id,
        SavedTalent.talent_id == request.talent_id
    ).first()

    #if not saved than saves
    if not saved:
        saved = SavedTalent(
            user_session_id=request.session_id or "direct_booking",
            user_id=request.user_id,
            talent_id=request.talent_id,
            saved_at=str(date.today())
        )
        db.add(saved)

    booking = Booking(     # Create Booking
        user_id=request.user_id,
        talent_id=request.talent_id,
        booking_date=str(datetime.now())
    )
    db.add(booking) #add booking
    db.commit()
    
    return {
        "status": "success",
        "message": f"Talent {talent.name} booked successfully."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8008, reload= True)
