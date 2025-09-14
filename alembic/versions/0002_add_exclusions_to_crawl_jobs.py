"""add exclusions column to crawl_jobs

Revision ID: 0002_add_exclusions
Revises: 0001_init
Create Date: 2025-09-12 00:00:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0002_add_exclusions'
down_revision = '0001_init'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('crawl_jobs', sa.Column('exclusions', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('crawl_jobs', 'exclusions')


