from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# Supabase provides postgres:// but SQLAlchemy requires postgresql://
db_url = settings.DATABASE_URL
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

connect_args = {}
engine_kwargs = {"pool_pre_ping": True}
if db_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False
else:
    # Managed Postgres (Render/Supabase) can silently drop idle connections;
    # recycle them before they go stale to avoid intermittent OperationalErrors.
    engine_kwargs["pool_recycle"] = 280

engine = create_engine(db_url, connect_args=connect_args, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
