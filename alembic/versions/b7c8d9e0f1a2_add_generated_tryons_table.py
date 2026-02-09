"""add generated tryons table

Revision ID: b7c8d9e0f1a2
Revises: a1b2c3d4e5f6
Create Date: 2026-02-10 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b7c8d9e0f1a2'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create generated_tryons table for virtual try-on results
    op.create_table(
        'generated_tryons',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('prompt', sa.Text(), nullable=False, server_default=''),
        sa.Column('clothing_image_key', sa.String(512), nullable=False),
        sa.Column('clothing_image_url', sa.Text(), nullable=True),
        sa.Column('result_image_key', sa.String(512), nullable=False),
        sa.Column('result_image_url', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )

    # Create index on user_id for faster queries
    op.create_index('ix_generated_tryons_user_id', 'generated_tryons', ['user_id'])


def downgrade() -> None:
    op.drop_index('ix_generated_tryons_user_id', table_name='generated_tryons')
    op.drop_table('generated_tryons')
