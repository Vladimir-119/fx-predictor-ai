"""Необязательные трейсы и метрики: только OTLP, без знания хранилищ."""

import logging
import math
import os
import socket
import traceback
from dataclasses import dataclass

import psutil
from opentelemetry import metrics as otel_metrics
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import Counter, Histogram, Meter, Observation, UpDownCounter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import OTELResourceDetector, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.config import Settings
from src.logging_config import redact_secrets

log = logging.getLogger(__name__)


@dataclass
class Instruments:
    """Имена и типы метрик общие для API, источника и воркеров."""

    api_requests: Counter
    api_duration: Histogram
    source_published: Counter
    source_rejections: Counter
    source_wait: Counter
    worker_tasks: Counter
    inference_duration: Histogram
    worker_active: UpDownCounter
    worker_redeliveries: Counter
    task_duration: Histogram


def create_instruments(meter: Meter) -> Instruments:
    """Идентификаторы задач и трейсов никогда не становятся лейблами метрик."""
    return Instruments(
        api_requests=meter.create_counter("fx_api_requests"),
        api_duration=meter.create_histogram("fx_api_duration", unit="s"),
        source_published=meter.create_counter("fx_source_published"),
        source_rejections=meter.create_counter("fx_source_rejections"),
        source_wait=meter.create_counter("fx_source_wait", unit="s"),
        worker_tasks=meter.create_counter("fx_worker_tasks"),
        inference_duration=meter.create_histogram("fx_inference_duration", unit="s"),
        worker_active=meter.create_up_down_counter("fx_worker_active"),
        worker_redeliveries=meter.create_counter("fx_worker_redeliveries"),
        task_duration=meter.create_histogram("fx_task_duration", unit="s"),
    )


metrics = create_instruments(otel_metrics.NoOpMeterProvider().get_meter(__name__))
_runtime: "Telemetry | None" = None


def get_tracer() -> trace.Tracer:
    """Возвращать наш tracer, не переопределяя глобальный provider MLflow."""
    if _runtime is not None:
        return _runtime.tracer
    return trace.NoOpTracerProvider().get_tracer(__name__)


def record_error(span: trace.Span, error: Exception) -> None:
    """Исключение в трейсе сохраняет тип и стек, скрывая DSN и токены."""
    span.record_exception(
        error,
        attributes={
            "exception.message": redact_secrets(str(error)),
            "exception.stacktrace": redact_secrets("".join(traceback.format_exception(error))),
        },
    )
    span.set_status(trace.Status(trace.StatusCode.ERROR, type(error).__name__))


def inference_buckets(settings: Settings) -> tuple[float, ...]:
    """Бакеты учитывают искусственную задержку, в том числе её отключение."""
    configured = os.getenv("OTEL_INFERENCE_HISTOGRAM_BUCKETS")
    if configured:
        values = tuple(float(value) for value in configured.split(","))
        if (
            not values
            or any(not math.isfinite(value) or value <= 0 for value in values)
            or tuple(sorted(set(values))) != values
        ):
            raise ValueError("OTEL_INFERENCE_HISTOGRAM_BUCKETS must be positive and increasing")
        return values
    delay = max(settings.inference_delay_seconds, 0.01)
    return tuple(
        sorted(
            {
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                *(delay * n for n in (0.5, 1, 1.01, 1.025, 1.05, 1.1, 1.5, 2, 4, 8)),
            }
        )
    )


class Telemetry:
    """Фоновые ограниченные очереди экспорта; остановка сбрасывает последний пакет."""

    def __init__(
        self,
        trace_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
    ) -> None:
        self.trace_provider = trace_provider
        self.meter_provider = meter_provider
        provider = trace_provider or trace.NoOpTracerProvider()
        self.tracer = provider.get_tracer("fx-predictor-ai")
        self.closed = False

    def flush(self) -> None:
        """Важно для источника, который завершается раньше следующего интервала."""
        if not self.closed:
            for provider in (self.meter_provider, self.trace_provider):
                if provider is not None:
                    try:
                        provider.force_flush(timeout_millis=1500)
                    except Exception as exc:  # noqa: BLE001 -- exporter must not break the pipeline
                        log.warning(
                            "telemetry_flush_failed", extra={"error_type": type(exc).__name__}
                        )

    def shutdown(self) -> None:
        """Вызывать после graceful shutdown обработки задач, не раньше."""
        global _runtime, metrics
        if self.closed:
            return
        self.flush()
        self.closed = True
        for provider in (self.trace_provider, self.meter_provider):
            if provider is not None:
                try:
                    provider.shutdown()
                except Exception as exc:  # noqa: BLE001 -- telemetry is an optional dependency
                    log.warning(
                        "telemetry_shutdown_failed", extra={"error_type": type(exc).__name__}
                    )
        if _runtime is self:
            _runtime = None
            metrics = create_instruments(otel_metrics.NoOpMeterProvider().get_meter(__name__))


def setup_telemetry(settings: Settings) -> Telemetry:
    """OTEL_SDK_DISABLED=true по умолчанию: обычный запуск не требует Collector."""
    global _runtime, metrics
    if os.getenv("OTEL_SDK_DISABLED", "true").lower() == "true":
        return Telemetry()
    if _runtime is not None:
        return _runtime
    resource = Resource.create(
        {
            "service.name": os.getenv("OTEL_SERVICE_NAME", settings.app_name),
            "service.version": os.getenv("OTEL_SERVICE_VERSION", settings.version),
            "service.instance.id": socket.gethostname(),
            "deployment.environment.name": settings.environment,
        }
    )
    resource = resource.merge(OTELResourceDetector().detect())
    bounds = inference_buckets(settings)
    timeout = float(os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT", "1"))
    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(timeout=timeout),
            max_queue_size=int(os.getenv("OTEL_BSP_MAX_QUEUE_SIZE", "512")),
            max_export_batch_size=int(os.getenv("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "64")),
            schedule_delay_millis=int(os.getenv("OTEL_BSP_SCHEDULE_DELAY", "1000")),
            export_timeout_millis=1500,
        )
    )
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(timeout=timeout),
        export_interval_millis=float(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "5000")),
        export_timeout_millis=1500,
    )
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[reader],
        views=[
            View(
                instrument_name="fx_inference_duration",
                aggregation=ExplicitBucketHistogramAggregation(bounds),
            ),
            View(
                instrument_name="fx_api_duration",
                aggregation=ExplicitBucketHistogramAggregation(
                    (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)
                ),
            ),
            View(
                instrument_name="fx_task_duration",
                aggregation=ExplicitBucketHistogramAggregation(
                    (0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120)
                ),
            ),
        ],
    )
    meter = meter_provider.get_meter("fx-predictor-ai")
    metrics = create_instruments(meter)
    process = psutil.Process()
    meter.create_observable_gauge(
        "process_memory_bytes",
        callbacks=[lambda _options: [Observation(process.memory_info().rss)]],
        unit="By",
    )
    meter.create_observable_gauge(
        "fx_queue_limit", callbacks=[lambda _options: [Observation(settings.queue_limit)]]
    )
    _runtime = Telemetry(trace_provider, meter_provider)
    return _runtime
