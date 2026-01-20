"""simplify user model

Revision ID: 599b6b884a1a
Revises:
Create Date: 2026-01-20 10:41:26.291209

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '599b6b884a1a'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop existing tables and recreate with new schema (dev only)
    op.drop_table('generated_images')
    op.drop_table('users')

    # Create users table with new simplified schema
    op.create_table(
        'users',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('reference_image_keys', sa.JSON(), nullable=False, default=[]),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )

    # Recreate generated_images table
    op.create_table(
        'generated_images',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('image_s3_key', sa.String(512), nullable=False),
        sa.Column('image_url', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    # Drop and recreate with old schema
    op.drop_table('generated_images')
    op.drop_table('users')

    op.create_table(
        'users',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('face_image_key', sa.String(512), nullable=False),
        sa.Column('upper_body_image_key', sa.String(512), nullable=True),
        sa.Column('full_image_key', sa.String(512), nullable=True),
        sa.Column('facial_attributes', sa.JSON(), nullable=True),
        sa.Column('face_embedding', sa.JSON(), nullable=True),
        sa.Column('face_quality_score', sa.Float(), nullable=True),
        sa.Column('face_detection_confidence', sa.Float(), nullable=True),
        sa.Column('original_images_count', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )

    op.create_table(
        'generated_images',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False),
        sa.Column('image_s3_key', sa.String(512), nullable=False),
        sa.Column('image_url', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )
