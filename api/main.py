import sys, os, uvicorn, logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Ensure GPU utilisation by default on compatible hardware
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestration.orchestrator import Orchestrator
from services.model_selector import RAGPipeline
from agents.base_agent import AgentNick
from agents.data_extraction_agent import DataExtractionAgent
from agents.supplier_ranking_agent import SupplierRankingAgent
from agents.quote_evaluation_agent import QuoteEvaluationAgent
from agents.quote_comparison_agent import QuoteComparisonAgent
from agents.opportunity_miner_agent import OpportunityMinerAgent
from agents.discrepancy_detection_agent import DiscrepancyDetectionAgent
from agents.email_drafting_agent import EmailDraftingAgent
from agents.negotiation_agent import NegotiationAgent
from agents.approvals_agent import ApprovalsAgent
from agents.supplier_interaction_agent import SupplierInteractionAgent
from services.email_watcher import SESEmailWatcher
from api.routers import workflows, system, run

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(LOG_DIR, "procwise.log"))])
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API starting up...")
    try:
        agent_nick = AgentNick()
        discrepancy_agent = DiscrepancyDetectionAgent(agent_nick)
        quote_evaluation_agent = QuoteEvaluationAgent(agent_nick)
        quote_comparison_agent = QuoteComparisonAgent(agent_nick)
        agent_nick.agents = {
            'data_extraction': DataExtractionAgent(agent_nick),
            'supplier_ranking': SupplierRankingAgent(agent_nick),
            'quote_evaluation': quote_evaluation_agent,
            'opportunity_miner': OpportunityMinerAgent(agent_nick),
            'DiscrepancyDetectionAgent': discrepancy_agent,
            'email_drafting': EmailDraftingAgent(agent_nick),
            'NegotiationAgent': NegotiationAgent(agent_nick),
            'ApprovalsAgent': ApprovalsAgent(agent_nick),
            'supplier_interaction': SupplierInteractionAgent(agent_nick),
            'QuoteEvaluationAgent': quote_evaluation_agent,
            'quote_comparison': quote_comparison_agent,
            'QuoteComparisonAgent': quote_comparison_agent,
        }
        agent_nick.email_watcher = SESEmailWatcher(
            agent_nick,
            supplier_agent=agent_nick.agents.get('supplier_interaction'),
            negotiation_agent=agent_nick.agents.get('NegotiationAgent'),
        )
        app.state.orchestrator = Orchestrator(agent_nick)
        app.state.rag_pipeline = RAGPipeline(agent_nick)
        logger.info("System initialized successfully.")
    except Exception as e:
        logger.critical(f"FATAL: System initialization failed: {e}", exc_info=True)
        app.state.orchestrator = None; app.state.rag_pipeline = None
    yield
    logger.info("API shutting down.")

app = FastAPI(title="ProcWise API v4 (Definitive)", version="4.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(workflows.router)
app.include_router(system.router)
app.include_router(run.router)

@app.get("/", tags=["General"])
def read_root(): return {"message": "Welcome to the ProcWise Agentic System API"}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
