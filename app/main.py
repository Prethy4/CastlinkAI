from datetime import date, datetime
from fastapi import FastAPI, HTTPException, Depends, Form
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage, AIMessage
from database import init_db, get_db, Talent, SavedTalent, ChatSession, ChatMessage, Booking, Draft
from schemas import ChatResponse, SaveTalentRequest, TalentResponse, ChatSessionResponse, BookTalentRequest, DraftResponse
from services import app_graph

from typing import List, Optional
import json
import os
# from dotenv import load_dotenv

# load_dotenv(override=True)

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

#########---------rag chat----------##########

# @app.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(
#     user_id: str = Form(...),
#     message: str = Form(...),
#     session_id: Optional[str] = Form(None),
#     location: Optional[str] = Form(None),
#     shoot_date: Optional[List[str]] = Form(None),
#     budget_range: Optional[int] = Form(None),
#     job_type: Optional[str] = Form(None),
#     db: Session = Depends(get_db)
#     ):
####################################################
# @app.post("/api/chat", response_model=ChatResponse)
# async def chat_endpoint(
#     request: ChatRequest, 
#     db: Session = Depends(get_db)
#     ):
#     try:
#         filters = {}
#         if request.optional_details:
#             if request.optional_details.location: filters['location'] = request.optional_details.location
#             if request.optional_details.job_type: filters['job_type'] = request.optional_details.job_type
#             if request.optional_details.shoot_dates: filters['shoot_dates'] = request.optional_details.shoot_dates
#             if request.optional_details.budget_range: filters['budget_range'] = request.optional_details.budget_range
        
#         chat_session = db.query(ChatSession).filter(ChatSession.user_id == request.user_id).order_by(ChatSession.created_at.desc()).first()
        
#         if not chat_session:
#             chat_session = ChatSession(user_id=request.user_id)
#             db.add(chat_session)
#             db.commit()
#             db.refresh(chat_session)

#         # Save User Message 
#         user_msg = ChatMessage(session_id=chat_session.id, sender="user", content=request.message)
#         db.add(user_msg)
#         db.commit()

#         past_messages = db.query(ChatMessage).filter(ChatMessage.session_id == chat_session.id).order_by(ChatMessage.id.desc()).limit(20).all()
#         past_messages.reverse()
#         msgs = []
#         for m in past_messages:
#             if m.sender == "user":
#                 msgs.append(HumanMessage(content=m.content))
#             else:
#                 msgs.append(AIMessage(content=m.content))

#         inputs = {
#             "messages": msgs,
#             "filters": filters
#         }
        
#         try:
#             final_state = app_graph.invoke(inputs, config={"recursion_limit": 20})
#             messages = final_state['messages']
#             last_msg = messages[-1]
#             response_content = last_msg.content
#         except Exception as e:
#                 raise e

#         # Save Response
#         ai_msg = ChatMessage(session_id=chat_session.id, sender="ai", content=response_content)
#         db.add(ai_msg)
#         db.commit()

#         suggested_talents = []
        
#         for msg in messages:
#             if hasattr(msg, 'name') and msg.name == 'generate_casting':
#                 try:
#                     data = json.loads(msg.content)
#                     if isinstance(data, list):
#                         for m in data:
#                             suggested_talents.append(TalentResponse(**m))
#                 except:
#                     pass

#         return ChatResponse(
#             session_id=chat_session.id,
#             response_text=response_content,
#             suggested_talents=suggested_talents
#         )
#     except Exception as e:
#         raise e

SERVER_START_TIME = str(datetime.now())

@app.post("/api/send", response_model=ChatResponse)
async def send_message(
    message: str = Form(..., description="Ask Anything"),
    user_id: str = Form(..., description="User ID"),
    location: Optional[str] = Form(None),
    shoot_dates: Optional[List[str]] = Form(None),
    budget_range: Optional[int] = Form(None),
    job_type: Optional[str] = Form(None),
    save_as_draft: bool = Form(False),
    db: Session = Depends(get_db)
    ):
    """Chat services for conversation with the chatbot"""
    chat_session = db.query(ChatSession).filter(ChatSession.user_id == user_id).order_by(ChatSession.created_at.desc()).first()
    
    if chat_session and chat_session.created_at < SERVER_START_TIME:
        chat_session = None
    
    if not chat_session:
        chat_session = ChatSession(user_id=user_id)
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)

    user_msg = ChatMessage(session_id=chat_session.id, sender="user", content=message)
    db.add(user_msg)
    db.commit()

    past_messages = db.query(ChatMessage).filter(ChatMessage.session_id == chat_session.id).order_by(ChatMessage.id.desc()).limit(10).all()
    past_messages.reverse()
    msgs = []
    for m in past_messages:
        if m.sender == "user":
            msgs.append(HumanMessage(content=m.content))
        else:
            msgs.append(AIMessage(content=m.content))

    filters = {}
    existing_draft = db.query(Draft).filter(Draft.session_id == chat_session.id).first()
    if existing_draft:
        if existing_draft.saved_filters: filters.update(existing_draft.saved_filters)

    if location: filters['location'] = location
    if shoot_dates:
        cleaned_dates = []
        for d in shoot_dates:
            if "," in d:
                cleaned_dates.extend([date.strip() for date in d.split(",")])
            else:
                cleaned_dates.append(d)
        filters['shoot_dates'] = cleaned_dates
    if budget_range: filters['budget_range'] = budget_range
    if job_type: filters['job_type'] = job_type

    inputs = {"messages": msgs, "filters": filters}
    
    final_state = app_graph.invoke(inputs, config={"recursion_limit": 5})
    
    if save_as_draft:
        current_filters = final_state.get('filters', {})
        
        mandatory_keys = ["gender", "category", "location", "job_type", "shoot_dates"]
        
        current_phase = "READY_TO_GENERATE" if all(k in current_filters for k in mandatory_keys) else "COLLECT_MANDATORY"

        draft = db.query(Draft).filter(Draft.session_id == chat_session.id).first()
        if not draft:
            draft = Draft(session_id=chat_session.id, user_id=user_id)
            db.add(draft)
        
        draft.phase = current_phase
        draft.saved_filters = current_filters
        draft.last_updated = str(datetime.now())
        db.commit()

    last_msg = final_state['messages'][-1]
    response_content = last_msg.content

    ai_msg = ChatMessage(session_id=chat_session.id, sender="ai", content=response_content)
    db.add(ai_msg)
    db.commit()

    suggested_talents = []
    for msg in final_state['messages']:
        if hasattr(msg, 'name') and msg.name == 'generate_casting':
            if hasattr(msg, 'artifact') and msg.artifact:
                for m in msg.artifact:
                    suggested_talents.append(TalentResponse(**m))
            else:
                try:
                    data = json.loads(msg.content)
                    if isinstance(data, list):
                        for m in data:
                            suggested_talents.append(TalentResponse(**m))
                except:
                    pass

    return ChatResponse(
        session_id=chat_session.id,
        response_text=response_content,
        suggested_talents=suggested_talents
    )

###########----------chat session-----------############

@app.get("/api/chat/sessions", response_model=List[ChatSessionResponse])
async def get_sessions(
    user_id: str, 
    db: Session = Depends(get_db)
    ):
    sessions = db.query(ChatSession).filter(ChatSession.user_id == user_id).all()
    return sessions

############----------get chat session by id--------############ recheck

@app.get("/api/chat/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_session_details(
    session_id: str, 
    db: Session = Depends(get_db)
    ):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@app.get("/api/chat/drafts/{session_id}", response_model=DraftResponse)
async def get_draft(
    session_id: str, 
    db: Session = Depends(get_db)
    ):
    draft = db.query(Draft).filter(Draft.session_id == session_id).first()
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    return draft

###########----------delete chat-----------############

@app.delete("/api/chat/sessions/{session_id}")
async def delete_session(
    session_id: str, 
    db: Session = Depends(get_db)
    ):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(session)
    db.commit()
    return {"status": "success", "message": "Session deleted"}

###########---------save a talent-------############

@app.post("/save-talent")
async def save_talents(
    request: SaveTalentRequest, 
    db: Session = Depends(get_db)
    ):
    """ 
    save a talent for further view
    """
    # Check if talent exists
    talent = db.query(Talent).filter(Talent.id == request.talent_id).first()
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

#########--------View calendar of a member that shows which dates they are available---------#########

@app.get("/talent/{talent_id}/availability")
async def get_availability(
    talent_id: str, 
    db: Session = Depends(get_db)
    ):
    """
    View availability dates of a member
    """
    talent = db.query(Talent).filter(Talent.id == talent_id).first()
    if not talent: raise HTTPException(404)
    return {
        "talent_id": talent.id,
        "name": talent.name,
        "available on": talent.availability
    }

# @app.get("/talent/{talent_id}/calendar")
# async def get_calendar(talent_id: str, db: Session = Depends(get_db)):
#     """
#     View calendar of a member that shows which dates they are available.
#     """
#     talent = db.query(Talent).filter(Talent.id == talent_id).first()
#     if not talent:
#         raise HTTPException(status_code=404, detail="Talent not found")
    
#     return {
#         "talent_id": talent.id,
#         "name": talent.name,
#         "availability": talent.availability
#     }

#########----------upload selftape endpoint---------############

@app.post("/talent/{talent_id}/selftape")
async def upload_selftape(
    talent_id: str
    ):
    """
    request for an selftape
    """
    return {
            "message": "Self-tape upload endpoint (add file handling here)"
            }

@app.get("/talent/{talent_id}/request_virtual_casting")
async def get_request_virtual_casting(
    talent_id: str, 
    db: Session = Depends(get_db)
    ):
    """
    request a virtual meet for casting
    """
    talent = db.query(Talent).filter(Talent.id == talent_id).first()
    if not talent: raise HTTPException(404)
    return {
            "virtual meet endpoint": talent.virtual_meet
            }

@app.post("/talent/{talent_id}/polas")
async def request_polas(
    talent_id: str
    ):
    """
    upload polas raw face images
    """
    return {
            "message": "polas request endpoint"
            }
    
@app.post("/book-talent")
async def book_talent(
    request: BookTalentRequest, 
    db: Session = Depends(get_db)
    ):
    """
    Book a talent. Automatically saves the talent if not already saved.
    """
    # To check if talent exists
    talent = db.query(Talent).filter(Talent.id == request.talent_id).first()
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
