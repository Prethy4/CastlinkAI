from sqlalchemy import create_engine, Column, String, Integer, BigInteger, JSON, ForeignKey, Date, Numeric, Boolean, Text, TIMESTAMP, CheckConstraint, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base, relationship 
from sqlalchemy.sql import func
from config import DATABASE_URL

Base = declarative_base()

class UserAuth(Base):
    __tablename__ = "accounts_user"
    
    user_id = Column(BigInteger, primary_key=True, index=True)
    full_name = Column(String, nullable=False)

class Talent(Base):
    __tablename__ = "talents"
    
    talent_id = Column(BigInteger, primary_key=True)
    name = Column(String, nullable=False) 
    role = Column(String, nullable=True)
    date_of_birth  = Column(Date, nullable=True)
    gender = Column(String(20), nullable=False)
    height = Column(String(50), nullable=True)
    bust = Column(String(50), nullable=True)
    waist = Column(String(50), nullable=True)
    hips = Column(String(50), nullable=True)
    shoe_size = Column(String(50), nullable=True)
    dress_size = Column(String(50), nullable=True)
    eye_colour = Column(String(100), nullable=False)
    hair_type = Column(String(100), nullable=False)
    hair_colour = Column(String(100), nullable=False)
    skin_color = Column(String(100), nullable=False)
    location = Column(String(255), nullable=False)
    continent = Column(String(100), nullable=False)
    country = Column(String(100), nullable=False)
    is_active = Column(Boolean, nullable=False)
    agent_id = Column(BigInteger, ForeignKey("accounts_user.user_id"), nullable=False)

    agent = relationship("UserAuth", backref="talents")
    images = relationship("TalentImage", back_populates="talent")
    available_dates = relationship("TalentAvailableDate", back_populates="talent")

class TalentAvailableDate(Base):
    __tablename__ = "talent_available_dates"
    
    availability_id = Column(BigInteger, primary_key=True, index=True)
    available_date = Column(Date, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    note = Column(String(255), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    talent_id = Column(BigInteger, ForeignKey("talents.talent_id"), nullable=False)

    talent = relationship("Talent", back_populates="available_dates")

class TalentImage(Base):
    __tablename__ = "talent_images"
    
    image_id = Column(BigInteger, primary_key=True)
    image = Column(String(100), nullable=False)
    talent_id = Column(BigInteger, ForeignKey("talents.talent_id"), nullable=False)
    talent = relationship("Talent", back_populates="images")

class ShortlistedTalent(Base):
    __tablename__ = "jobs_shortlisted_talents"
    
    shortlisted_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.session_id"), nullable=True, index=True)
    talent_id = Column(BigInteger, ForeignKey("talents.talent_id"), nullable=False)
    user_id = Column(BigInteger, ForeignKey("accounts_user.user_id"), nullable=False)
    job_id = Column(Integer, ForeignKey("jobs_talent_job.job_id"), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    talent = relationship("Talent")
    user = relationship("UserAuth")
    job = relationship("Job")
    session = relationship("ChatSession")

class Booking(Base):
    __tablename__ = "jobs_talent_bookings"
    
    booking_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.session_id"), nullable=True, index=True)
    user_id = Column(BigInteger, ForeignKey("accounts_user.user_id"), nullable=False)
    talent_id = Column(BigInteger, ForeignKey("talents.talent_id"), nullable=False)
    job_id = Column(Integer, ForeignKey("jobs_talent_job.job_id"), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    talent = relationship("Talent")
    user = relationship("UserAuth")
    job = relationship("Job")
    session = relationship("ChatSession")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    session_id = Column(String, primary_key=True, default=lambda: __import__('uuid').uuid4().hex)
    user_id = Column(BigInteger)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    draft = relationship("Draft", back_populates="session", uselist=False, cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    message_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.session_id"))
    sender = Column(String)
    content = Column(String)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    location = Column(String, nullable=True)
    shoot_date = Column(String, nullable=True)
    budget = Column(String, nullable=True)
    job_type = Column(String, nullable=True)
    session = relationship("ChatSession", back_populates="messages")

class Draft(Base):
    __tablename__ = "jobs_talent_drafts"
    
    draft_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(BigInteger)
    session_id = Column(String, ForeignKey("chat_sessions.session_id"), unique=True, index=True)
    phase = Column(String)
    saved_filters = Column(JSON)
    last_updated = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    job_type = Column(String, nullable=True)
    location = Column(String, nullable=True)
    shoot_date = Column(String, nullable=True)
    budget = Column(String, nullable=True)

    session = relationship("ChatSession", back_populates="draft")

class Job(Base):
    __tablename__ = "jobs_talent_job"   
    
    job_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.session_id"), nullable=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    location = Column(String(255), nullable=False)
    budget_min = Column(Numeric(10, 2))
    budget_max = Column(Numeric(10, 2))
    job_type = Column(String(120), nullable=False)
    status = Column(String(20), nullable=False)
    applicants_count = Column(Integer, nullable=False, default=0)
    shortlisted_count = Column(Integer, nullable=False, default=0)
    selftapes_count = Column(Integer, nullable=False, default=0)
    ecastings_count = Column(Integer, nullable=False, default=0)
    polas_count = Column(Integer, nullable=False, default=0)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    # job_assigned_to_id = Column(
    #     BigInteger,
    #     ForeignKey("accounts_user.user_id"),
    #     nullable=True
    # )
    job_created_by_id = Column(
        BigInteger,
        ForeignKey("accounts_user.user_id"),
        nullable=False
    )

    session = relationship("ChatSession", back_populates="jobs")

    @property
    def budget(self):
        if self.budget_min is None and self.budget_max is None:
            return None
        
        def fmt(val):
            if val is None: return "0"
            return f"{int(val)}" if val % 1 == 0 else f"{val:.2f}"

        if self.budget_min == self.budget_max:
            return f"{fmt(self.budget_min)}$"
        return f"{fmt(self.budget_min)}-{fmt(self.budget_max)}$"

    __table_args__ = (
        CheckConstraint('applicants_count >= 0'),
        CheckConstraint('shortlisted_count >= 0'),
        CheckConstraint('selftapes_count >= 0'),
        CheckConstraint('ecastings_count >= 0'),
        CheckConstraint('polas_count >= 0'),
    )

class JobAIResult(Base):
    __tablename__ = "jobs_ai_results"
    
    result_id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs_talent_job.job_id"), nullable=False)
    suggested_talents = Column(JSON)
    requested_selftapes = Column(JSON, nullable=True)
    requested_ecastings = Column(JSON, nullable=True)
    requested_polas = Column(JSON, nullable=True)
    shoot_date = Column(String, nullable=True)

# Setup Engine
if "sqlite" in DATABASE_URL:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    migrate_db()

def migrate_db():
    """Helper to add columns that create_all doesn't handle for existing tables."""
    inspector = inspect(engine)
    with engine.begin() as conn:
        # Update jobs_talent_job table for polas_count
        job_cols = [c['name'] for c in inspector.get_columns("jobs_talent_job")]
        if "polas_count" not in job_cols:
            conn.execute(text("ALTER TABLE jobs_talent_job ADD COLUMN polas_count INTEGER DEFAULT 0 NOT NULL"))
        
        # Update jobs_ai_results table for requested_polas
        ai_cols = [c['name'] for c in inspector.get_columns("jobs_ai_results")]
        if "requested_polas" not in ai_cols:
            conn.execute(text("ALTER TABLE jobs_ai_results ADD COLUMN requested_polas JSONB"))
            
        # Update talent_available_dates table for note
        avail_cols = [c['name'] for c in inspector.get_columns("talent_available_dates")]
        if "note" not in avail_cols:
            conn.execute(text("ALTER TABLE talent_available_dates ADD COLUMN note VARCHAR(255)"))
            
        # Update jobs_shortlisted_talents for missing columns
        if "jobs_shortlisted_talents" in inspector.get_table_names():
            short_cols = [c['name'] for c in inspector.get_columns("jobs_shortlisted_talents")]
            if "session_id" not in short_cols:
                conn.execute(text("ALTER TABLE jobs_shortlisted_talents ADD COLUMN session_id VARCHAR REFERENCES chat_sessions(session_id)"))
            if "job_id" not in short_cols:
                conn.execute(text("ALTER TABLE jobs_shortlisted_talents ADD COLUMN job_id INTEGER REFERENCES jobs_talent_job(job_id)"))
        
        # Update jobs_talent_bookings for missing columns
        if "jobs_talent_bookings" in inspector.get_table_names():
            booking_cols = [c['name'] for c in inspector.get_columns("jobs_talent_bookings")]
            if "session_id" not in booking_cols:
                conn.execute(text("ALTER TABLE jobs_talent_bookings ADD COLUMN session_id VARCHAR REFERENCES chat_sessions(session_id)"))
            if "job_id" not in booking_cols:
                conn.execute(text("ALTER TABLE jobs_talent_bookings ADD COLUMN job_id INTEGER REFERENCES jobs_talent_job(job_id)"))

    """
    ### SQL MIGRATION SCRIPTS FOR NEONDB ###
    
    -- 1. Add missing columns to existing tables
    ALTER TABLE jobs_talent_job ADD COLUMN IF NOT EXISTS polas_count INTEGER DEFAULT 0 NOT NULL;
    ALTER TABLE jobs_ai_results ADD COLUMN IF NOT EXISTS requested_polas JSONB;
    ALTER TABLE talent_available_dates ADD COLUMN IF NOT EXISTS note VARCHAR(255);

    -- 2. Create Shortlisted Talents table if not exists
    CREATE TABLE IF NOT EXISTS jobs_shortlisted_talents (
        shortlisted_id SERIAL PRIMARY KEY,
        talent_id BIGINT NOT NULL REFERENCES talents(talent_id),
        user_id BIGINT NOT NULL REFERENCES accounts_user(user_id),
        job_id INTEGER REFERENCES jobs_talent_job(job_id),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    -- 3. Create Bookings table if not exists
    CREATE TABLE IF NOT EXISTS jobs_talent_bookings (
        booking_id SERIAL PRIMARY KEY,
        user_id BIGINT NOT NULL REFERENCES accounts_user(user_id),
        talent_id BIGINT NOT NULL REFERENCES talents(talent_id),
        job_id INTEGER REFERENCES jobs_talent_job(job_id),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()