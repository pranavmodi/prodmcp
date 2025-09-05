import os
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker


# Default to local Postgres per user request
# user=neha password=postgres db=postgres host=localhost port=5432
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://neha:postgres@localhost:5432/postgres",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

Base = declarative_base()


class CrawlJob(Base):
    __tablename__ = "crawl_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(128), nullable=False, index=True)
    url = Column(Text, nullable=False)
    status = Column(String(32), nullable=False, default="started")
    message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    finished_at = Column(DateTime(timezone=True), nullable=True)


def create_crawl_job(tenant_id: str, url: str, status: str = "started") -> int:
    with SessionLocal() as session:
        job = CrawlJob(tenant_id=tenant_id, url=url, status=status)
        session.add(job)
        session.commit()
        session.refresh(job)
        return int(job.id)


def set_crawl_status(job_id: int, status: str, message: Optional[str] = None, finished: bool = False) -> None:
    if not job_id:
        return
    with SessionLocal() as session:
        job = session.get(CrawlJob, job_id)
        if not job:
            return
        job.status = status
        if message is not None:
            job.message = message
        if finished:
            job.finished_at = func.now()
        session.commit()


