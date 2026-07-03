"""
tracing - Arize Phoenix + Opentelemetry instrumentation bootstrap.
Call init_tracing() once at startup before any LangGraph invocations.
"""
import logging
from pathlib import Path
import phoenix as px 
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from src.config import PHOENIX_ENDPOINT, PHOENIX_PROJECT_NAME

logger = logging.getLogger(__name__)

_initialized = False 

def init_tracing() -> None:
    """Start Pheonix locally and configure OpenTelemetry."""
    global _initialized
    if _initialized:
        return 

    import os
    os.environ["PHOENIX_PROJECT_NAME"] = PHOENIX_PROJECT_NAME
    # Point Phoenix to a permanent directory so traces survive server restarts.
    # Without this, launch_app() defaults to use_temp_dir=True which writes to
    # a temp folder that is wiped when the Python process exits.
    os.environ.setdefault("PHOENIX_WORKING_DIR", str(Path.home() / ".phoenix"))

    try:
        px.launch_app(use_temp_dir=False)
        logger.info("Pheonix launched at http://localhost:6006")
    except Exception as e:
        logger.info("Phoenix app might already be running: %s", e)

    tracer_provider = trace_sdk.TracerProvider()
    
    span_exporter = OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT)
    span_processor = SimpleSpanProcessor(span_exporter)
    tracer_provider.add_span_processor(span_processor)

    trace_api.set_tracer_provider(tracer_provider)

    LangChainInstrumentor().instrument()

    _initialized = True 
    logger.info("Tracing initialized: OTEL -> Phoenix at %s", PHOENIX_ENDPOINT)

def get_tracer(name: str = PHOENIX_PROJECT_NAME):
    """Get a named tracer for custom spans."""
    return trace_api.get_tracer(name)
    