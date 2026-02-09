"""add generated videos table

Revision ID: a1b2c3d4e5f6
Revises: 599b6b884a1a
Create Date: 2026-01-28 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '599b6b884a1a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create generated_videos table
    op.create_table(
        'generated_videos',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('video_s3_key', sa.String(512), nullable=False),
        sa.Column('video_url', sa.Text(), nullable=True),
        sa.Column('source_type', sa.String(20), nullable=False, server_default='text'),
        sa.Column('source_image_id', sa.String(36), sa.ForeignKey('generated_images.id', ondelete='SET NULL'), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )

    # Create index on user_id for faster queries
    op.create_index('ix_generated_videos_user_id', 'generated_videos', ['user_id'])


def downgrade() -> None:
    op.drop_index('ix_generated_videos_user_id', table_name='generated_videos')
    op.drop_table('generated_videos')
