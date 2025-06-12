"""
Unit tests for Streaming Simulation (Subtask 3.6).

Tests cover:
- Real-time event streaming
- Kafka integration
- Event ordering and processing
- Backpressure handling
- Stream aggregation
- Performance under load
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
from unittest.mock import MagicMock, AsyncMock

from src.ingestion.streaming import (
    StreamProcessor,
    EventStream,
    KafkaConsumer,
    StreamAggregator,
    EventBuffer,
    StreamMetrics
)
from src.db.models.identity import Identity
from src.db.models.audit import IdentityEvent, EventType
from src.db.models.saas_application import SaaSProvider


class TestStreamingSimulation:
    """Test suite for streaming simulation functionality."""

    @pytest.fixture
    def stream_processor(self):
        """Create a stream processor instance."""
        return StreamProcessor(
            buffer_size=1000,
            flush_interval=1.0,
            max_batch_size=100
        )

    @pytest.fixture
    def mock_kafka_consumer(self, mocker):
        """Create a mock Kafka consumer."""
        consumer = mocker.MagicMock()
        consumer.subscribe = mocker.MagicMock()
        consumer.poll = mocker.MagicMock()
        consumer.commit = mocker.MagicMock()
        return consumer

    @pytest.mark.asyncio
    async def test_basic_event_streaming(self, stream_processor):
        """Test basic event streaming functionality."""
        events_processed = []

        async def process_event(event):
            events_processed.append(event)

        stream_processor.on_event(process_event)

        # Simulate events
        for i in range(10):
            event = {
                "event_id": f"evt_{i}",
                "event_type": "user.created",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"user_id": f"user_{i}"}
            }
            await stream_processor.process_event(event)

        # Allow processing
        await asyncio.sleep(0.1)

        assert len(events_processed) == 10
        assert all(e["event_id"].startswith("evt_") for e in events_processed)

    @pytest.mark.asyncio
    async def test_event_ordering(self, stream_processor):
        """Test that events are processed in order."""
        processed_order = []

        async def track_order(event):
            processed_order.append(event["sequence"])

        stream_processor.on_event(track_order)

        # Send events with sequence numbers
        events = []
        for i in range(100):
            event = {
                "event_id": f"evt_{i}",
                "sequence": i,
                "timestamp": datetime.utcnow().isoformat()
            }
            events.append(event)

        # Process all events
        for event in events:
            await stream_processor.process_event(event)

        await stream_processor.flush()

        # Verify order preserved
        assert processed_order == list(range(100))

    @pytest.mark.asyncio
    async def test_batch_processing(self, stream_processor):
        """Test batch processing of events."""
        batches_received = []

        async def process_batch(batch):
            batches_received.append(len(batch))

        stream_processor.on_batch(process_batch)
        stream_processor.max_batch_size = 10

        # Send 25 events
        for i in range(25):
            event = {"event_id": f"evt_{i}"}
            await stream_processor.process_event(event)

        await stream_processor.flush()

        # Should receive 3 batches: 10, 10, 5
        assert len(batches_received) == 3
        assert batches_received[0] == 10
        assert batches_received[1] == 10
        assert batches_received[2] == 5

    @pytest.mark.asyncio
    async def test_backpressure_handling(self, stream_processor):
        """Test backpressure handling when processing is slow."""
        slow_processing_time = 0.1
        events_dropped = []

        async def slow_processor(event):
            await asyncio.sleep(slow_processing_time)

        async def drop_handler(event):
            events_dropped.append(event)

        stream_processor.on_event(slow_processor)
        stream_processor.on_drop(drop_handler)
        stream_processor.buffer_size = 5  # Small buffer

        # Send more events than buffer can handle
        for i in range(20):
            try:
                await stream_processor.process_event(
                    {"event_id": f"evt_{i}"},
                    timeout=0.01  # Short timeout
                )
            except asyncio.TimeoutError:
                pass

        # Some events should be dropped
        assert len(events_dropped) > 0

    @pytest.mark.asyncio
    async def test_kafka_integration(self, mock_kafka_consumer):
        """Test Kafka consumer integration."""
        stream = EventStream(consumer=mock_kafka_consumer)

        # Mock Kafka messages
        mock_messages = [
            MagicMock(
                value=json.dumps({
                    "event_id": f"kafka_evt_{i}",
                    "event_type": "user.updated",
                    "data": {"user_id": f"user_{i}"}
                }).encode()
            )
            for i in range(5)
        ]

        mock_kafka_consumer.poll.return_value = {
            "topic": mock_messages
        }

        events_received = []

        async def collect_events(event):
            events_received.append(event)

        stream.on_event(collect_events)

        # Start consuming
        consume_task = asyncio.create_task(stream.consume())

        # Let it process
        await asyncio.sleep(0.1)

        # Stop consuming
        stream.stop()
        await consume_task

        assert len(events_received) == 5
        assert all(e["event_id"].startswith("kafka_evt_") for e in events_received)

    @pytest.mark.asyncio
    async def test_stream_aggregation(self):
        """Test stream aggregation functionality."""
        aggregator = StreamAggregator(
            window_size=timedelta(seconds=1),
            aggregation_function="count"
        )

        # Send events for different users
        for i in range(100):
            event = {
                "event_type": "page.view",
                "user_id": f"user_{i % 10}",  # 10 unique users
                "timestamp": datetime.utcnow()
            }
            aggregator.add_event(event)

        # Get aggregated results
        results = aggregator.get_results()

        assert len(results) == 10  # One per unique user
        assert sum(r["count"] for r in results) == 100

    @pytest.mark.asyncio
    async def test_event_deduplication(self, stream_processor):
        """Test duplicate event detection and handling."""
        unique_events = []

        async def process_unique(event):
            unique_events.append(event)

        stream_processor.on_event(process_unique)
        stream_processor.enable_deduplication(window_size=100)

        # Send duplicate events
        for _ in range(3):
            for i in range(5):
                event = {
                    "event_id": f"dup_evt_{i}",
                    "data": {"value": i}
                }
                await stream_processor.process_event(event)

        await stream_processor.flush()

        # Should only process unique events
        assert len(unique_events) == 5

    @pytest.mark.asyncio
    async def test_stream_filtering(self, stream_processor):
        """Test event filtering in streams."""
        filtered_events = []

        async def collect_filtered(event):
            filtered_events.append(event)

        # Add filter for specific event types
        stream_processor.add_filter(
            lambda e: e.get("event_type") in ["user.created", "user.deleted"]
        )
        stream_processor.on_event(collect_filtered)

        # Send various event types
        event_types = ["user.created", "user.updated", "user.deleted", "user.login"]
        for i, event_type in enumerate(event_types * 5):
            event = {
                "event_id": f"evt_{i}",
                "event_type": event_type
            }
            await stream_processor.process_event(event)

        await stream_processor.flush()

        # Only created and deleted events should pass
        assert all(
            e["event_type"] in ["user.created", "user.deleted"]
            for e in filtered_events
        )

    @pytest.mark.asyncio
    async def test_stream_transformation(self, stream_processor):
        """Test event transformation in streams."""
        transformed_events = []

        async def collect_transformed(event):
            transformed_events.append(event)

        # Add transformation
        def transform_event(event):
            return {
                **event,
                "processed_at": datetime.utcnow().isoformat(),
                "provider": event.get("source", "unknown").upper()
            }

        stream_processor.add_transformer(transform_event)
        stream_processor.on_event(collect_transformed)

        # Send events
        event = {
            "event_id": "transform_1",
            "source": "okta"
        }
        await stream_processor.process_event(event)

        await stream_processor.flush()

        assert len(transformed_events) == 1
        assert "processed_at" in transformed_events[0]
        assert transformed_events[0]["provider"] == "OKTA"

    @pytest.mark.asyncio
    async def test_stream_metrics(self, stream_processor):
        """Test stream metrics collection."""
        metrics = StreamMetrics()
        stream_processor.attach_metrics(metrics)

        # Process various events
        for i in range(50):
            event = {
                "event_id": f"metric_evt_{i}",
                "event_type": "user.created" if i % 2 == 0 else "user.updated"
            }
            await stream_processor.process_event(event)

        await stream_processor.flush()

        # Check metrics
        stats = metrics.get_stats()
        assert stats["total_events"] == 50
        assert stats["events_per_type"]["user.created"] == 25
        assert stats["events_per_type"]["user.updated"] == 25
        assert stats["avg_processing_time"] > 0

    @pytest.mark.asyncio
    async def test_stream_recovery(self, stream_processor):
        """Test stream recovery after failure."""
        processed_before_failure = []
        processed_after_recovery = []

        async def process_with_failure(event):
            if event["event_id"] == "evt_5":
                raise Exception("Simulated failure")
            processed_before_failure.append(event["event_id"])

        async def process_after_recovery(event):
            processed_after_recovery.append(event["event_id"])

        stream_processor.on_event(process_with_failure)

        # Send events until failure
        for i in range(10):
            try:
                await stream_processor.process_event({"event_id": f"evt_{i}"})
            except Exception:
                # Simulate recovery
                stream_processor.clear_handlers()
                stream_processor.on_event(process_after_recovery)

                # Replay from last checkpoint
                for j in range(5, 10):
                    await stream_processor.process_event({"event_id": f"evt_{j}"})

        await stream_processor.flush()

        # Verify recovery
        assert "evt_5" not in processed_before_failure
        assert len(processed_after_recovery) == 5

    @pytest.mark.asyncio
    async def test_multi_stream_join(self):
        """Test joining multiple event streams."""
        stream1 = EventStream(name="users")
        stream2 = EventStream(name="permissions")

        joined_events = []

        async def join_handler(user_event, perm_event):
            joined_events.append({
                "user_id": user_event["user_id"],
                "permission": perm_event["permission"]
            })

        # Create stream joiner
        joiner = StreamJoiner(
            stream1, stream2,
            join_key="user_id",
            window=timedelta(seconds=1)
        )
        joiner.on_join(join_handler)

        # Send events to both streams
        for i in range(5):
            await stream1.emit({
                "user_id": f"user_{i}",
                "action": "login"
            })
            await stream2.emit({
                "user_id": f"user_{i}",
                "permission": "read"
            })

        await joiner.process()

        assert len(joined_events) == 5

    @pytest.mark.asyncio
    async def test_stream_performance(self, stream_processor):
        """Test stream processing performance."""
        import time

        events_processed = 0

        async def count_events(event):
            nonlocal events_processed
            events_processed += 1

        stream_processor.on_event(count_events)

        start_time = time.time()

        # Send large number of events
        for i in range(10000):
            event = {
                "event_id": f"perf_evt_{i}",
                "timestamp": datetime.utcnow().isoformat()
            }
            await stream_processor.process_event(event)

        await stream_processor.flush()

        end_time = time.time()
        duration = end_time - start_time

        # Should process 10K events quickly
        assert events_processed == 10000
        assert duration < 5.0  # Under 5 seconds

        # Calculate throughput
        throughput = events_processed / duration
        assert throughput > 2000  # At least 2000 events/second

    @pytest.mark.asyncio
    async def test_stream_error_recovery_strategies(self, stream_processor):
        """Test different error recovery strategies."""
        attempts = {}

        async def process_with_retry(event):
            event_id = event["event_id"]
            attempts[event_id] = attempts.get(event_id, 0) + 1

            if attempts[event_id] < 3:
                raise Exception(f"Temporary failure for {event_id}")

            return f"Processed {event_id}"

        stream_processor.on_event(process_with_retry)
        stream_processor.set_retry_policy(
            max_retries=3,
            backoff_factor=0.1
        )

        # Send event that will fail twice
        result = await stream_processor.process_event({"event_id": "retry_test"})

        assert attempts["retry_test"] == 3
        assert result == "Processed retry_test"
