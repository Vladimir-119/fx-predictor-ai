"""Экспорт заменён памятью: контекст, ресурсы, метрики и отключение без Docker."""

import contextvars
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest
from opentelemetry import propagate, trace
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src import logging_config, telemetry


def test_disabled_telemetry_never_creates_exporters(settings, monkeypatch):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setattr(telemetry, "_runtime", None)

    def unexpected_exporter(**kwargs):
        raise AssertionError("Disabled telemetry must not create an exporter")

    monkeypatch.setattr(telemetry, "OTLPSpanExporter", unexpected_exporter)
    monkeypatch.setattr(telemetry, "OTLPMetricExporter", unexpected_exporter)
    runtime = telemetry.setup_telemetry(settings)
    with telemetry.get_tracer().start_as_current_span("disabled") as span:
        assert not span.is_recording()
    runtime.flush()
    runtime.shutdown()
    runtime.shutdown()
    assert not trace.get_current_span().get_span_context().is_valid


def test_export_resources_traceparent_thread_context_logs_and_metrics(settings, monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "fx-inference-worker")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "always_on")
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "service.version=demo-v2,deployment.environment.name=defense,service.instance.id=worker-2",
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "0.5")
    monkeypatch.setenv("OTEL_METRIC_EXPORT_INTERVAL", "3000")
    exporter = InMemorySpanExporter()
    reader = InMemoryMetricReader()
    exporter_options = {}

    def trace_exporter(**kwargs):
        exporter_options.update(kwargs)
        return exporter

    def metric_reader(exporter, **kwargs):
        assert kwargs["export_interval_millis"] == 3000
        return reader

    monkeypatch.setattr(telemetry, "OTLPSpanExporter", trace_exporter)
    monkeypatch.setattr(telemetry, "OTLPMetricExporter", lambda **kwargs: object())
    monkeypatch.setattr(telemetry, "PeriodicExportingMetricReader", metric_reader)
    runtime = telemetry.setup_telemetry(settings)
    try:
        assert telemetry.setup_telemetry(settings) is runtime
        tracer = telemetry.get_tracer()
        with tracer.start_as_current_span("publish") as publisher:
            carrier = {}
            propagate.inject(carrier)
            assert carrier["traceparent"].startswith("00-")
            trace_id = publisher.get_span_context().trace_id
        with tracer.start_as_current_span("process", context=propagate.extract(carrier)) as process:
            with logging_config.bind_task("task-123"):
                with logging_config.bind_task("nested"):
                    assert logging_config.TASK_ID.get() == "nested"
                assert logging_config.TASK_ID.get() == "task-123"
                context = contextvars.copy_context()

                def inference():
                    with tracer.start_as_current_span("inference") as infer:
                        record = logging.LogRecord(
                            "model", logging.INFO, __file__, 1, "inferred", (), None
                        )
                        parsed = json.loads(logging_config.JsonFormatter().format(record))
                        assert parsed["task_id"] == "task-123"
                        assert parsed["trace_id"] == format(trace_id, "032x")
                        assert parsed["span_id"] == format(infer.get_span_context().span_id, "016x")
                        assert parsed["service"] == "fx-inference-worker"
                        return infer.get_span_context()

                with ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(context.run, inference).result(timeout=2)
                assert result.trace_id == trace_id
                assert process.get_span_context().trace_id == trace_id
        assert logging_config.TASK_ID.get() is None
        telemetry.metrics.api_requests.add(1, {"route": "/api/v1/tasks", "status_code": 503})
        telemetry.metrics.api_duration.record(0.025, {"route": "/api/v1/tasks"})
        telemetry.metrics.inference_duration.record(settings.inference_delay_seconds + 0.01)
        telemetry.metrics.task_duration.record(0.5)
        telemetry.metrics.worker_active.add(1)
        telemetry.metrics.worker_active.add(-1)
        telemetry.metrics.source_published.add(1)
        telemetry.metrics.source_rejections.add(1)
        telemetry.metrics.source_wait.add(0.1)
        telemetry.metrics.worker_tasks.add(1, {"result": "ok"})
        telemetry.metrics.worker_redeliveries.add(1)
        data = reader.get_metrics_data()
        resource = data.resource_metrics[0].resource.attributes
        assert resource["service.name"] == "fx-inference-worker"
        assert resource["service.version"] == "demo-v2"
        assert resource["service.instance.id"] == "worker-2"
        assert resource["deployment.environment.name"] == "defense"
        values = {
            item.name: item
            for scope in data.resource_metrics[0].scope_metrics
            for item in scope.metrics
        }
        assert values["fx_api_requests"].data.data_points[0].value == 1
        assert values["process_memory_bytes"].data.data_points[0].value > 0
        assert values["fx_queue_limit"].data.data_points[0].value == settings.queue_limit
        buckets = values["fx_inference_duration"].data.data_points[0].explicit_bounds
        assert buckets == telemetry.inference_buckets(settings)
        runtime.flush()
        spans = exporter.get_finished_spans()
        by_name = {span.name: span for span in spans}
        assert by_name["process"].parent.span_id == by_name["publish"].context.span_id
        assert by_name["inference"].parent.span_id == by_name["process"].context.span_id
        assert exporter_options["timeout"] == 0.5
    finally:
        runtime.shutdown()
    assert telemetry._runtime is None
    assert not telemetry.get_tracer().start_span("after-shutdown").is_recording()


def test_histogram_buckets_follow_configuration(settings, monkeypatch):
    settings.inference_delay_seconds = 0
    zero_delay = telemetry.inference_buckets(settings)
    settings.inference_delay_seconds = 2
    assert max(telemetry.inference_buckets(settings)) > max(zero_delay)
    monkeypatch.setenv("OTEL_INFERENCE_HISTOGRAM_BUCKETS", "0.01,0.1,1,3")
    assert telemetry.inference_buckets(settings) == (0.01, 0.1, 1, 3)
    for invalid in ("0,1", "1,0.5", "1,1", "nan,1", "inf"):
        monkeypatch.setenv("OTEL_INFERENCE_HISTOGRAM_BUCKETS", invalid)
        with pytest.raises(ValueError, match="positive and increasing"):
            telemetry.inference_buckets(settings)


def test_errors_hide_secrets_and_export_failure_does_not_stop_shutdown(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "always_on")
    from types import SimpleNamespace
    from unittest.mock import Mock

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        with provider.get_tracer("test").start_as_current_span("failure") as span:
            try:
                raise ValueError("postgresql://user:hidden@postgres/db token=private")
            except ValueError as exc:
                telemetry.record_error(span, exc)
        recorded = exporter.get_finished_spans()[0]
        assert recorded.status.status_code is trace.StatusCode.ERROR
        attributes = recorded.events[0].attributes
        assert "hidden" not in attributes["exception.message"]
        assert "private" not in attributes["exception.stacktrace"]
        assert "ValueError" in attributes["exception.type"]
    finally:
        provider.shutdown()
    runtime = telemetry.Telemetry()
    runtime.trace_provider = SimpleNamespace(
        force_flush=Mock(side_effect=TimeoutError("collector down")),
        shutdown=Mock(side_effect=OSError("collector down")),
    )
    runtime.meter_provider = SimpleNamespace(
        force_flush=Mock(side_effect=TimeoutError("collector down")),
        shutdown=Mock(side_effect=OSError("collector down")),
    )
    runtime.shutdown()
    assert runtime.closed
    runtime.trace_provider.shutdown.assert_called_once()
    runtime.meter_provider.shutdown.assert_called_once()
