"""
Admin Panel Router

Provides endpoints for admin authentication and session management.
Includes filtering, sorting, and export capabilities.
"""

import logging
import csv
import io
from fastapi import APIRouter, HTTPException, Depends, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional
from datetime import datetime

from myapp.models.hr_models import (
    AdminLogin,
    AdminToken,
    SessionSummary,
    RecruiterSession
)
from myapp.utils.auth import verify_password, create_admin_token, get_current_admin
from myapp.utils.session_store import get_session_store

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/admin",
    tags=["Admin Panel"]
)

session_store = get_session_store()


@router.post("/login", response_model=AdminToken)
async def login(request: Request, credentials: AdminLogin):
    """
    Admin login endpoint.
    
    Authenticates admin user and returns JWT token.
    Rate limited to 5 attempts per 15 minutes per IP.
    
    Args:
        credentials: Admin password
        
    Returns:
        JWT access token and expiration time
    """
    try:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Verify password
        if not verify_password(credentials.password, client_ip):
            raise HTTPException(
                status_code=401,
                detail="Invalid password"
            )
        
        # Create token
        token, expires_at = create_admin_token(client_ip)
        
        return AdminToken(
            access_token=token,
            token_type="bearer",
            expires_at=expires_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Login failed"
        )


@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions(
    company: Optional[str] = Query(None, description="Filter by company (partial match)"),
    role: Optional[str] = Query(None, description="Filter by role (partial match)"),
    start_date: Optional[datetime] = Query(None, description="Filter sessions after this date"),
    end_date: Optional[datetime] = Query(None, description="Filter sessions before this date"),
    interest_level: Optional[str] = Query(None, description="Filter by interest level"),
    sort_by: str = Query("date", description="Sort field (date/company/interest_level/duration)"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    admin: dict = Depends(get_current_admin)
):
    """
    List all recruiter sessions with filtering and sorting.
    
    Requires admin authentication.
    
    Args:
        company: Filter by company name (case-insensitive partial match)
        role: Filter by role (case-insensitive partial match)
        start_date: Filter sessions after this date
        end_date: Filter sessions before this date
        interest_level: Filter by interest level (low/medium/high)
        sort_by: Sort field
        sort_order: Sort order (asc/desc)
        admin: Admin user from JWT token
        
    Returns:
        List of session summaries
    """
    try:
        logger.info(f"Admin listing sessions (filters: company={company}, role={role})")
        
        # Load all sessions
        sessions = await session_store.list_sessions()
        
        # Apply filters
        filtered_sessions = []
        
        for session in sessions:
            # Skip sessions without metadata or analysis
            if not session.metadata or not session.analysis:
                continue
            
            # Company filter
            if company and company.lower() not in session.recruiter_info.company.lower():
                continue
            
            # Role filter
            if role:
                session_role = session.recruiter_info.role or ""
                if role.lower() not in session_role.lower():
                    continue
            
            # Date filters
            if start_date and session.metadata.start_time < start_date:
                continue
            
            if end_date and session.metadata.start_time > end_date:
                continue
            
            # Interest level filter
            if interest_level and session.analysis.interest_level != interest_level.lower():
                continue
            
            filtered_sessions.append(session)
        
        # Convert to summaries
        summaries = [
            SessionSummary(
                session_id=session.session_id,
                recruiter_name=session.recruiter_info.name,
                recruiter_email=session.recruiter_info.email,
                company=session.recruiter_info.company,
                role=session.recruiter_info.role,
                start_time=session.metadata.start_time,
                session_duration_seconds=session.metadata.session_duration_seconds,
                interest_level=session.analysis.interest_level,
                total_messages=session.metadata.total_messages
            )
            for session in filtered_sessions
        ]
        
        # Sort
        sort_key_map = {
            "date": lambda s: s.start_time,
            "company": lambda s: s.company.lower(),
            "interest_level": lambda s: s.interest_level,
            "duration": lambda s: s.session_duration_seconds
        }
        
        sort_key = sort_key_map.get(sort_by, sort_key_map["date"])
        reverse = (sort_order.lower() == "desc")
        
        summaries.sort(key=sort_key, reverse=reverse)
        
        logger.info(f"Returning {len(summaries)} sessions")
        
        return summaries
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list sessions: {str(e)}"
        )


@router.get("/session/{session_id}", response_model=RecruiterSession)
async def get_session(
    session_id: str,
    admin: dict = Depends(get_current_admin)
):
    """
    Get complete details of a recruiter session.
    
    Requires admin authentication.
    
    Args:
        session_id: Session identifier
        admin: Admin user from JWT token
        
    Returns:
        Complete RecruiterSession object
    """
    try:
        logger.info(f"Admin viewing session: {session_id}")
        
        # Load session
        session = await session_store.load_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}"
            )
        
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session: {str(e)}"
        )


@router.get("/export/{session_id}")
async def export_session(
    session_id: str,
    admin: dict = Depends(get_current_admin)
):
    """
    Export a single session as JSON file.
    
    Requires admin authentication.
    
    Args:
        session_id: Session identifier
        admin: Admin user from JWT token
        
    Returns:
        JSON file download
    """
    try:
        logger.info(f"Admin exporting session: {session_id}")
        
        # Load session
        session = await session_store.load_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}"
            )
        
        # Convert to JSON
        session_json = session.model_dump_json(indent=2)
        
        # Return as downloadable file
        return StreamingResponse(
            io.BytesIO(session_json.encode()),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={session_id}.json"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export session: {str(e)}"
        )


@router.get("/export_all")
async def export_all_sessions(
    format: str = Query("json", description="Export format (json/csv)"),
    company: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    interest_level: Optional[str] = Query(None),
    admin: dict = Depends(get_current_admin)
):
    """
    Export all sessions as JSON or CSV.
    
    Supports same filtering as list_sessions endpoint.
    Requires admin authentication.
    
    Args:
        format: Export format (json or csv)
        company: Filter by company
        role: Filter by role
        start_date: Filter by start date
        end_date: Filter by end date
        interest_level: Filter by interest level
        admin: Admin user from JWT token
        
    Returns:
        JSON array or CSV file download
    """
    try:
        logger.info(f"Admin exporting all sessions (format: {format})")
        
        # Get filtered sessions (reuse list_sessions logic)
        summaries = await list_sessions(
            company=company,
            role=role,
            start_date=start_date,
            end_date=end_date,
            interest_level=interest_level,
            sort_by="date",
            sort_order="desc",
            admin=admin
        )
        
        # Load full sessions
        sessions = []
        for summary in summaries:
            session = await session_store.load_session(summary.session_id)
            if session:
                sessions.append(session)
        
        if format.lower() == "csv":
            # Generate CSV
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "Session ID",
                "Recruiter Name",
                "Recruiter Email",
                "Company",
                "Role",
                "Start Time",
                "End Time",
                "Duration (seconds)",
                "Total Messages",
                "Interest Level",
                "Conversation Summary"
            ])
            
            # Write rows
            for session in sessions:
                if not session.metadata or not session.analysis:
                    continue
                
                writer.writerow([
                    session.session_id,
                    session.recruiter_info.name,
                    session.recruiter_info.email,
                    session.recruiter_info.company,
                    session.recruiter_info.role or "",
                    session.metadata.start_time.isoformat(),
                    session.metadata.end_time.isoformat(),
                    session.metadata.session_duration_seconds,
                    session.metadata.total_messages,
                    session.analysis.interest_level,
                    session.analysis.conversation_summary
                ])
            
            # Return CSV
            output.seek(0)
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=recruiter_sessions_{datetime.utcnow().strftime('%Y%m%d')}.csv"
                }
            )
        
        else:
            # Return JSON
            sessions_json = [session.model_dump(mode='json') for session in sessions]
            
            return JSONResponse(
                content=sessions_json,
                headers={
                    "Content-Disposition": f"attachment; filename=recruiter_sessions_{datetime.utcnow().strftime('%Y%m%d')}.json"
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export sessions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export sessions: {str(e)}"
        )


@router.get("/")
async def admin_home(admin: dict = Depends(get_current_admin)):
    """Admin panel health check."""
    session_count = await session_store.get_session_count()
    
    return {
        "service": "Admin Panel",
        "status": "active",
        "admin_role": admin.get("role"),
        "total_sessions": session_count
    }
