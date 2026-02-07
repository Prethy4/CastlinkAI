import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, BigInteger, JSON, ForeignKey, Date
from sqlalchemy.orm import sessionmaker, declarative_base, relationship 
from config import DATABASE_URL

Base = declarative_base()

class Talent(Base):
    __tablename__ = "talents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())) 
    name = Column(String)
    gender = Column(String)
    category = Column(String)
    agent_name = Column(String)
    virtual_meet = Column(String)
    height = Column(String)
    bust = Column(String)
    waist = Column(String)
    hips = Column(String)
    dress_size = Column(String)
    shoe_size = Column(String)
    hair = Column(String)
    hair_type = Column(String)
    eyes = Column(String)
    skin_color = Column(String)
    location = Column(String)
    continent = Column(String)
    country = Column(String)
    job_types = Column(JSON) 
    availability = Column(String) 
    photos = Column(String) 
    bio = Column(String) 
    shoot_dates= Column(Date)
    budget_tier = Column(BigInteger)

class SavedTalent(Base):
    __tablename__ = "saved_talents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_session_id = Column(String, index=True)
    talent_id = Column(String, ForeignKey("talents.id"))
    saved_at = Column(String)
    user_id = Column(String)

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String)
    created_at = Column(String, default=lambda: str(datetime.now()))
    
    messages = relationship("ChatMessage", back_populates="session")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    sender = Column(String)
    content = Column(String)
    timestamp = Column(String, default=lambda: str(datetime.now()))
    
    session = relationship("ChatSession", back_populates="messages")

class Booking(Base):
    __tablename__ = "bookings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String)
    talent_id = Column(String, ForeignKey("talents.id"))
    booking_date = Column(String)

class Draft(Base):
    __tablename__ = "drafts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String)
    session_id = Column(String, ForeignKey("chat_sessions.id"), unique=True, index=True)
    phase = Column(String)
    saved_filters = Column(JSON)
    last_updated = Column(String)

# Setup Engine
if "sqlite" in DATABASE_URL:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=1800)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()