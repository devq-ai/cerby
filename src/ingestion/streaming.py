"""
Streaming processor for Cerby Identity Automation Platform.

This module implements Kinesis-like streaming capabilities for processing
high-volume identity events in real-time with partitioning, checkpointing,
and replay functionality.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import hashlib
import uuid
import logfire

from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from src.ingestion.base import BaseIngestionHandler, IngestionResult, IngestionStatus, IngestionError
from src.db.models.identity import Identity, IdentityProvider
from src.db.models.saas_app import SaaSApplication
from src.db.models.audit import IdentityEvent, EventType
from src.core.config import settings


class StreamStatus(str, Enum):
    """Status of a stream processor."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StreamPartition:
    """Represents a stream partition."""
    partition_id: int
    current_offset: int = 0
    last_checkpoint: datetime = field(default_factory=datetime.utcnow)
    events_processed: int = 0
    events_failed: int = 0
    lag: int = 0


@dataclass
class StreamEvent:
    """Represents an event in the stream."""
    event_id: str
    partition_key: str
    sequence_number: int
    timestamp: datetime
    data: Dict[str, Any]
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "partition_key": self.partition_key,
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "attributes": self.attributes
        }


class StreamProcessor(BaseIngestionHandler):
    """
    Processes identity events from a streaming source.

    Implements Kinesis-like functionality including:
    - Multiple partitions for parallel processing
    - Checkpointing for fault tolerance
    - Event replay capability
    - Backpressure handling
    - Dead letter queue for failed events
    """

    def __init__(self, db_session: AsyncSession, saas_app: SaaSApplication,
                 num_partitions: int = 4, buffer_size: int = 1000):
        super().__init__(db_session, saas_app)

        self.num_partitions = num_partitions
        self.buffer_size = buffer_size

        # Stream components
        self.partitions: Dict[int, StreamPartition] = {}
        self.event_buffers: Dict[int, deque] = {}
        self.checkpoint_interval = timedelta(seconds=30)
        self.max_batch_size = 100

        # Processing state
        self.status = StreamStatus.IDLE
        self.processors: List[asyncio.Task] = []
        self.producer_task: Optional[asyncio.Task] = None

        # Metrics
        self.metrics = {
            "events_received": 0,
            "events_processed": 0,
            "events_failed": 0,
            "bytes_processed": 0,
            "processing_time_ms": deque(maxlen=1000),
            "throughput_events_per_sec": 0.0
        }

        # Redis for distributed processing (optional)
        self.redis_client: Optional[redis.Redis] = None
        self.use_redis = bool(settings.redis_url)

        # Dead letter queue
        self.dlq: deque = deque(maxlen=10000)

        # Event handlers
        self.event_handlers: Dict[str, Callable] = {}

        # Initialize partitions
        self._initialize_partitions()

    def _initialize_partitions(self) -> None:
        """Initialize stream partitions."""
        for i in range(self.num_partitions):
            self.partitions[i] = StreamPartition(partition_id=i)
            self.event_buffers[i] = deque(maxlen=self.buffer_size)

    async def ingest(self, **kwargs) -> IngestionResult:
        """
        Start stream processing.

        This method starts the stream processor and returns immediately.
        Use stop() to stop processing and get final results.
        """
        if self.status == StreamStatus.RUNNING:
            raise IngestionError("Stream processor already running")

        self.result.status = IngestionStatus.IN_PROGRESS
        self.status = StreamStatus.RUNNING

        with logfire.span("Start stream processor", provider=self.provider):
            try:
                # Connect to Redis if configured
                if self.use_redis:
                    await self._connect_redis()

                # Start partition processors
                for partition_id in range(self.num_partitions):
                    processor = asyncio.create_task(
                        self._process_partition(partition_id)
                    )
                    self.processors.append(processor)

                # Start metrics collector
                asyncio.create_task(self._collect_metrics())

                # Start checkpoint manager
                asyncio.create_task(self._checkpoint_manager())

                logfire.info(
                    "Stream processor started",
                    provider=self.provider,
                    partitions=self.num_partitions
                )

            except Exception as e:
                self.status = StreamStatus.ERROR
                self.result.status = IngestionStatus.FAILED
                self.result.add_error(str(e))
                raise

        return self.result

    async def produce_event(self, identity_data: Dict[str, Any],
                          event_type: str = "identity.update") -> None:
        """
        Produce an event to the stream.

        Args:
            identity_data: Identity data to process
            event_type: Type of event
        """
        if self.status != StreamStatus.RUNNING:
            raise IngestionError("Stream processor not running")

        # Create stream event
        event = StreamEvent(
            event_id=str(uuid.uuid4()),
            partition_key=identity_data.get("external_id", str(uuid.uuid4())),
            sequence_number=int(time.time() * 1000000),  # Microsecond timestamp
            timestamp=datetime.utcnow(),
            data=identity_data,
            attributes={"event_type": event_type}
        )

        # Determine partition
        partition_id = self._get_partition_for_key(event.partition_key)

        # Add to buffer
        self.event_buffers[partition_id].append(event)
        self.metrics["events_received"] += 1

        # Update partition lag
        self.partitions[partition_id].lag = len(self.event_buffers[partition_id])

        # Store in Redis if configured
        if self.redis_client:
            await self._store_event_redis(event, partition_id)

    async def stop(self) -> IngestionResult:
        """Stop stream processing and return results."""
        if self.status not in [StreamStatus.RUNNING, StreamStatus.PAUSED]:
            return self.result

        with logfire.span("Stop stream processor", provider=self.provider):
            self.status = StreamStatus.STOPPING

            # Cancel all processors
            for processor in self.processors:
                processor.cancel()

            # Wait for processors to finish
            await asyncio.gather(*self.processors, return_exceptions=True)

            # Final checkpoint
            await self._checkpoint_all_partitions()

            # Disconnect from Redis
            if self.redis_client:
                await self.redis_client.close()

            self.status = StreamStatus.STOPPED
            self.result.complete()

            logfire.info(
                "Stream processor stopped",
                provider=self.provider,
                events_processed=self.metrics["events_processed"],
                events_failed=self.metrics["events_failed"]
            )

        return self.result

    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate streaming event data."""
        return "external_id" in data and "email" in data

    async def transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform streaming event data."""
        # Add stream metadata
        data["stream_metadata"] = {
            "processed_at": datetime.utcnow().isoformat(),
            "partition_id": data.get("_partition_id"),
            "sequence_number": data.get("_sequence_number")
        }
        return data

    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """Register a custom event handler."""
        self.event_handlers[event_type] = handler

    async def replay_from_timestamp(self, start_time: datetime,
                                   end_time: Optional[datetime] = None) -> None:
        """
        Replay events from a specific timestamp.

        Args:
            start_time: Start timestamp for replay
            end_time: End timestamp (current time if not specified)
        """
        if self.status == StreamStatus.RUNNING:
            raise IngestionError("Cannot replay while processor is running")

        end_time = end_time or datetime.utcnow()

        with logfire.span("Replay events", provider=self.provider,
                         start_time=start_time, end_time=end_time):
            # Query events from database
            from sqlalchemy import select
            stmt = select(IdentityEvent).where(
                IdentityEvent.provider == self.provider,
                IdentityEvent.occurred_at >= start_time,
                IdentityEvent.occurred_at <= end_time
            ).order_by(IdentityEvent.occurred_at)

            result = await self.db_session.execute(stmt)
            events = result.scalars().all()

            # Process events
            for event in events:
                identity_data = event.event_data.get("identity_data", {})
                await self.produce_event(identity_data, event.event_type)

            logfire.info(
                "Event replay completed",
                provider=self.provider,
                events_replayed=len(events)
            )

    async def _process_partition(self, partition_id: int) -> None:
        """Process events from a specific partition."""
        partition = self.partitions[partition_id]
        buffer = self.event_buffers[partition_id]

        with logfire.span("Process partition", partition_id=partition_id):
            while self.status == StreamStatus.RUNNING:
                try:
                    # Check for events
                    if not buffer:
                        await asyncio.sleep(0.1)
                        continue

                    # Process batch
                    batch = []
                    while buffer and len(batch) < self.max_batch_size:
                        batch.append(buffer.popleft())

                    if batch:
                        await self._process_event_batch(batch, partition_id)
                        partition.lag = len(buffer)

                except Exception as e:
                    logfire.error(
                        "Partition processing error",
                        partition_id=partition_id,
                        error=str(e)
                    )
                    # Move events to DLQ
                    for event in batch:
                        self.dlq.append({
                            "event": event.to_dict(),
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        })

    async def _process_event_batch(self, events: List[StreamEvent],
                                  partition_id: int) -> None:
        """Process a batch of events."""
        start_time = time.time()
        partition = self.partitions[partition_id]

        for event in events:
            try:
                # Add partition metadata
                event.data["_partition_id"] = partition_id
                event.data["_sequence_number"] = event.sequence_number

                # Get event handler
                event_type = event.attributes.get("event_type", "identity.update")
                handler = self.event_handlers.get(event_type)

                if handler:
                    # Use custom handler
                    await handler(event.data)
                else:
                    # Use default processing
                    if await self.validate_data(event.data):
                        identity_data = await self.transform_data(event.data)
                        await self.process_identity(identity_data)

                partition.events_processed += 1
                self.metrics["events_processed"] += 1
                self.metrics["bytes_processed"] += len(json.dumps(event.data))

            except Exception as e:
                partition.events_failed += 1
                self.metrics["events_failed"] += 1

                # Add to DLQ
                self.dlq.append({
                    "event": event.to_dict(),
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })

        # Update metrics
        processing_time_ms = (time.time() - start_time) * 1000
        self.metrics["processing_time_ms"].append(processing_time_ms)

        # Update partition offset
        if events:
            partition.current_offset = events[-1].sequence_number

    async def _checkpoint_manager(self) -> None:
        """Periodically checkpoint partition progress."""
        while self.status == StreamStatus.RUNNING:
            await asyncio.sleep(self.checkpoint_interval.total_seconds())
            await self._checkpoint_all_partitions()

    async def _checkpoint_all_partitions(self) -> None:
        """Checkpoint all partitions."""
        for partition_id, partition in self.partitions.items():
            await self._checkpoint_partition(partition_id)

    async def _checkpoint_partition(self, partition_id: int) -> None:
        """Save partition checkpoint."""
        partition = self.partitions[partition_id]

        checkpoint_data = {
            "partition_id": partition_id,
            "offset": partition.current_offset,
            "events_processed": partition.events_processed,
            "events_failed": partition.events_failed,
            "timestamp": datetime.utcnow().isoformat()
        }

        if self.redis_client:
            # Store in Redis
            key = f"stream:checkpoint:{self.provider}:{partition_id}"
            await self.redis_client.set(
                key,
                json.dumps(checkpoint_data),
                ex=86400  # 24 hour expiry
            )

        partition.last_checkpoint = datetime.utcnow()

        logfire.debug(
            "Partition checkpoint saved",
            partition_id=partition_id,
            offset=partition.current_offset
        )

    async def _collect_metrics(self) -> None:
        """Collect and calculate metrics."""
        last_event_count = 0
        last_time = time.time()

        while self.status == StreamStatus.RUNNING:
            await asyncio.sleep(5)  # Update every 5 seconds

            current_time = time.time()
            current_event_count = self.metrics["events_processed"]

            # Calculate throughput
            time_delta = current_time - last_time
            event_delta = current_event_count - last_event_count

            if time_delta > 0:
                self.metrics["throughput_events_per_sec"] = event_delta / time_delta

            # Calculate average processing time
            if self.metrics["processing_time_ms"]:
                avg_processing_time = sum(self.metrics["processing_time_ms"]) / len(self.metrics["processing_time_ms"])
            else:
                avg_processing_time = 0

            # Log metrics
            logfire.info(
                "Stream metrics",
                provider=self.provider,
                throughput=self.metrics["throughput_events_per_sec"],
                avg_processing_time_ms=avg_processing_time,
                events_processed=current_event_count,
                events_failed=self.metrics["events_failed"],
                total_lag=sum(p.lag for p in self.partitions.values())
            )

            last_event_count = current_event_count
            last_time = current_time

    def _get_partition_for_key(self, partition_key: str) -> int:
        """Determine partition for a given key."""
        # Use consistent hashing
        hash_value = int(hashlib.md5(partition_key.encode()).hexdigest(), 16)
        return hash_value % self.num_partitions

    async def _connect_redis(self) -> None:
        """Connect to Redis for distributed processing."""
        self.redis_client = await redis.from_url(
            settings.redis_url,
            password=settings.redis_password,
            decode_responses=True
        )
        await self.redis_client.ping()
        logfire.info("Connected to Redis for stream processing")

    async def _store_event_redis(self, event: StreamEvent, partition_id: int) -> None:
        """Store event in Redis for distributed processing."""
        if not self.redis_client:
            return

        # Store in partition-specific list
        key = f"stream:partition:{self.provider}:{partition_id}"
        await self.redis_client.rpush(key, json.dumps(event.to_dict()))

        # Trim to max size
        await self.redis_client.ltrim(key, -self.buffer_size, -1)

    async def get_partition_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics for all partitions."""
        stats = {}

        for partition_id, partition in self.partitions.items():
            stats[partition_id] = {
                "current_offset": partition.current_offset,
                "events_processed": partition.events_processed,
                "events_failed": partition.events_failed,
                "lag": partition.lag,
                "last_checkpoint": partition.last_checkpoint.isoformat()
            }

        return stats

    async def get_dlq_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get events from dead letter queue."""
        return list(self.dlq)[-limit:]

    async def reprocess_dlq(self) -> int:
        """Reprocess events from dead letter queue."""
        reprocessed = 0
        failed = []

        while self.dlq:
            dlq_entry = self.dlq.popleft()
            event_data = dlq_entry["event"]["data"]

            try:
                if await self.validate_data(event_data):
                    identity_data = await self.transform_data(event_data)
                    await self.process_identity(identity_data)
                    reprocessed += 1
            except Exception as e:
                failed.append(dlq_entry)

        # Put failed events back
        self.dlq.extend(failed)

        return reprocessed
