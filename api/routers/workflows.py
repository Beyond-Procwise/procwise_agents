# ProcWise/api/routers/workflows.py

import json
import os

from pydantic import BaseModel

from orchestration.orchestrator import Orchestrator
from services.model_selector import RAGPipeline

class RankingRequest(BaseModel):
    query: str

class ExtractRequest(BaseModel):
    s3_prefix: Optional[str] = None
    s3_object_key: Optional[str] = None


class AgentType(BaseModel):
    agentId: int
    agentType: str
    description: str
    dependencies: List[str]


def load_from_json(file_path: str):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500, detail=f"Configuration file not found: {file_path}"
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500, detail=f"Error decoding JSON from file: {file_path}"
        )


router = APIRouter(prefix="/workflows", tags=["Agent Workflows"])



@router.post("/ask")
async def ask_question(
    params: QuestionParams = Depends(),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    if len(params.files) > 3:
        raise HTTPException(400, "Max 3 files allowed.")
    file_data = [(await f.read(), f.filename) for f in params.files if f.filename]



@router.get(
    "/types",
    response_model=List[AgentType],
    summary="Get agent types and their resource dependencies",
)
def get_agent_types():
    """
    Returns a list of all defined agent types, their descriptions,
    and their dependencies from the agent_definitions.json file.
    """

