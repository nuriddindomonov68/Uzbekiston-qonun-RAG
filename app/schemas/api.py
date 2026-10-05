from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class SessionCreate(BaseModel):
    session_id: Optional[str] = None
    settings:   Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    settings:   Dict[str, Any]


class ChatRequest(BaseModel):
    message:    str  = Field(..., description="User query")
    session_id: str  = Field(..., description="Active session ID")


class ChatResponse(BaseModel):
    response:    str
    session_id:  str
    dataset_id:  Optional[str] = None
    chart_paths: Optional[List[str]] = None


class DatasetResponse(BaseModel):
    dataset_id:   str
    filename:     str
    filepath:     str
    file_type:    str
    row_count:    int
    column_count: int
    schema_info:  Dict[str, str]
    uploaded_at:  str


class ModelResponse(BaseModel):
    model_id:      str
    session_id:    str
    dataset_id:    str
    model_type:    str
    algorithm:     str
    target_column: Optional[str]
    features:      List[str]
    metrics:       Dict[str, Any]
    trained_at:    str
