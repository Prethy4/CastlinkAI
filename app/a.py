import os
import jwt
from dotenv import load_dotenv
from config import JWT_SECRET_KEY
 
load_dotenv()
 
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzc4MDQwNDU4LCJpYXQiOjE3NzU0NDg0NTgsImp0aSI6IjA5OTgxNWJjZjFlMTRjODFhMWEzOTU0YzQ2NDhkMmYxIiwidXNlcl9pZCI6IjkifQ.Lma0JIjTGrEBNP-pOA2Nm8NazaKifrlnsGJgVqU_awM"
secret = JWT_SECRET_KEY
 
decoded = jwt.decode(token, secret, algorithms=["HS256"])
print(decoded)
 
 
#  from fastapi.responses import JSONResponse
# from sqlalchemy.orm import Session
# from langchain_core.messages import HumanMessage, AIMessage
# from database import init_db, get_db, ChatSession, ChatMessage, Draft, Job, JobAIResult, UserAuth, Talent
# from schemas import TalentResponse, ChatSessionResponse, DraftResponse, ChatRequest, JobResponse, ContinueDraftResponse, ChatMessageResponse, JobResultResponse, WrappedChatResponse, PaginationResponse, TalentDataResponse, UserDraftResponse, DraftsSavedFilters, RequestTalentJobRequest
# from services import app_graph, extract_information, generate_ask_response, CustomEncoder, RateLimiter, time_ago, parse_budget, generate_job_details_from_messages
# from typing import List, Optional
# import json

#     ai_result = db.query(JobAIResult).filter(JobAIResult.job_id == job.job_id).first()
#     suggested_talents = ai_result.suggested_talents if ai_result else []
#     requested_selftapes_raw = ai_result.requested_selftapes if ai_result else []
#     requested_ecastings_raw = ai_result.requested_ecastings if ai_result else []
#     shoot_date = ai_result.shoot_date if ai_result else None

#     messages = []
#     response = JobResultResponse(
#         **job.__dict__,
#         shoot_date=shoot_date,
#         suggested_talents=[TalentResponse(**t) for t in (suggested_talents or [])],
#         requested_selftapes=[TalentResponse(**t) for t in (requested_selftapes_raw or [])],
#         requested_ecastings=[TalentResponse(**t) for t in (requested_ecastings_raw or [])],
#         messages=[ChatMessageResponse.from_orm(m) for m in messages]
#     )
#     return response

# @app.post("/api/jobs/request-selftape", dependencies=[Depends(limiter)])
# async def request_selftape(
#     request: RequestTalentJobRequest,
#     user_id: int = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Endpoint to request a self-tape for a specific talent in a job."""
#     job = db.query(Job).filter(Job.job_id == request.job_id, Job.job_created_by_id == user_id).first()
#     if not job:
#         raise HTTPException(status_code=404, detail="Job not found")

#     talent = db.query(Talent).filter(Talent.talent_id == request.talent_id).first()
#     if not talent:
#         raise HTTPException(status_code=404, detail="Talent not found")

#     # Increment the self-tape count for the job
#     job.selftapes_count += 1

#     # Retrieve or create JobAIResult to store the snapshot
#     ai_result = db.query(JobAIResult).filter(JobAIResult.job_id == job.job_id).first()
#     if not ai_result:
#         ai_result = JobAIResult(job_id=job.job_id, requested_selftapes=[])
#         db.add(ai_result)
    
#     selftapes_list = ai_result.requested_selftapes or []
    
#     # Add talent snapshot if not already in the list
#     if not any(t.get('talent_id') == talent.talent_id for t in selftapes_list):
#         talent_snapshot = {
#             "talent_id": talent.talent_id,
#             "name": talent.name,
#             "role": talent.role,
#             "gender": talent.gender,
#             "location": talent.location,
#             "country": talent.country,
#             "continent": talent.continent,
#             "is_active": talent.is_active,
#             "agent_id": talent.agent_id,
#             "agent_name": talent.agent.full_name if talent.agent else "Unknown",
#             "images": [f"/media/{img.image}" for img in sorted(talent.images, key=lambda x: x.image_id)[:1]] if talent.images else [],
#             "eye_color": talent.eye_colour,
#             "hair_type": talent.hair_type,
#             "hair_color": talent.hair_colour,
#             "skin_color": talent.skin_color,
#             "height": talent.height, "bust": talent.bust, "waist": talent.waist, "hips": talent.hips,
#             "shoe_size": talent.shoe_size, "dress_size": talent.dress_size
#         }
#         selftapes_list.append(talent_snapshot)
#         ai_result.requested_selftapes = selftapes_list

#     db.commit()
#     return {"status": "success", "message": f"Self-tape requested for {talent.name}"}

# @app.post("/api/jobs/request-ecasting", dependencies=[Depends(limiter)])
# async def request_ecasting(
#     request: RequestTalentJobRequest,
#     user_id: int = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Endpoint to request an e-casting for a specific talent in a job."""
#     job = db.query(Job).filter(Job.job_id == request.job_id, Job.job_created_by_id == user_id).first()
#     if not job:
#         raise HTTPException(status_code=404, detail="Job not found")

#     talent = db.query(Talent).filter(Talent.talent_id == request.talent_id).first()
#     if not talent:
#         raise HTTPException(status_code=404, detail="Talent not found")

#     # Increment the e-casting count for the job
#     job.ecastings_count += 1

#     ai_result = db.query(JobAIResult).filter(JobAIResult.job_id == job.job_id).first()
#     if not ai_result:
#         ai_result = JobAIResult(job_id=job.job_id, requested_ecastings=[])
#         db.add(ai_result)
    
#     ecastings_list = ai_result.requested_ecastings or []
    
#     if not any(t.get('talent_id') == talent.talent_id for t in ecastings_list):
#         talent_snapshot = {
#             "talent_id": talent.talent_id,
#             "name": talent.name,
#             "role": talent.role,
#             "gender": talent.gender,
#             "location": talent.location,
#             "country": talent.country,
#             "continent": talent.continent,
#             "is_active": talent.is_active,
#             "agent_id": talent.agent_id,
#             "agent_name": talent.agent.full_name if talent.agent else "Unknown",
#             "images": [f"/media/{img.image}" for img in sorted(talent.images, key=lambda x: x.image_id)[:1]] if talent.images else [],
#             "eye_color": talent.eye_colour,
#             "hair_type": talent.hair_type,
#             "hair_color": talent.hair_colour,
#             "skin_color": talent.skin_color,
#             "height": talent.height, "bust": talent.bust, "waist": talent.waist, "hips": talent.hips,
#             "shoe_size": talent.shoe_size, "dress_size": talent.dress_size
#         }
#         ecastings_list.append(talent_snapshot)
#         ai_result.requested_ecastings = ecastings_list

#     db.commit()
#     return {"status": "success", "message": f"E-casting requested for {talent.name}"}

# # ##########---------save a talent-------############ 
