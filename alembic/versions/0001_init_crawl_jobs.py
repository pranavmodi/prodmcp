"""init crawl_jobs

Revision ID: 0001_init
Revises: 
Create Date: 2025-09-05 00:00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0001_init'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'crawl_jobs',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('tenant_id', sa.String(length=128), nullable=False, index=True),
        sa.Column('url', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='started'),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('idx_crawl_jobs_tenant', 'crawl_jobs', ['tenant_id'])


def downgrade() -> None:
    op.drop_index('idx_crawl_jobs_tenant', table_name='crawl_jobs')
    op.drop_table('crawl_jobs')


