import logging
import os

import torch

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from services.email_service import EmailService

logger = logging.getLogger(__name__)

# Ensure GPU variables are set
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_DEFAULT_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)
if torch.cuda.is_available():  # pragma: no cover - hardware dependent
    torch.set_default_device("cuda")
else:  # pragma: no cover - hardware dependent
    logger.warning("CUDA not available; defaulting to CPU.")


class EmailDraftingAgent(BaseAgent):
    """Agent that drafts an email body using an LLM and sends it via SES."""

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.email_service = EmailService(agent_nick)

    def run(self, context: AgentContext) -> AgentOutput:
        subject = context.input_data.get("subject", "")
        prompt = context.input_data.get("prompt")
        body = context.input_data.get("body")
        recipient = context.input_data.get("recipient")
        sender = context.input_data.get(
            "sender", self.agent_nick.settings.ses_default_sender
        )

        if not recipient:
            return AgentOutput(
                status=AgentStatus.FAILED,
                data={},
                error="recipient not provided",
            )

        if prompt and not body:
            response = self.call_ollama(prompt=prompt)
            body = response.get("response", "")

        success = self.email_service.send_email(subject, body or "", recipient, sender)
        status = AgentStatus.SUCCESS if success else AgentStatus.FAILED
        return AgentOutput(status=status, data={"subject": subject, "body": body, "recipient": recipient, "sender": sender})
