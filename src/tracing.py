"""
tracing - Arize Phoenix + Opentelemetry instrumentation bootstrap.
Call init_tracing() once at startup before any LangGraph invocations.
"""
import logging 
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

    px.launch_app()
    logger.info("Pheonix launched at http://localhost:6006")

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
    