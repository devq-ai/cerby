"""
Batch import handler for Cerby Identity Automation Platform.

This module handles bulk identity imports from CSV and JSON files,
supporting large file processing with streaming and validation.
"""

import csv
import json
import io
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime
import chardet
import aiofiles
from dataclasses import dataclass
import pandas as pd
import logfire

from fastapi import UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.ingestion.base import BaseIngestionHandler, IngestionResult, IngestionStatus, IngestionError
from src.db.models.identity import IdentityProvider
from src.db.models.saas_app import SaaSApplication
from src.core.config import settings


@dataclass
class BatchImportConfig:
    """Configuration for batch import operations."""

    # File processing
    chunk_size: int = 1000
    max_file_size_mb: int = 100
    encoding: Optional[str] = None

    # CSV specific
    delimiter: str = ","
    quotechar: str = '"'
    has_header: bool = True

    # Field mapping
    field_mapping: Dict[str, str] = None
    required_fields: List[str] = None

    # Validation
    skip_invalid_rows: bool = True
    validate_email: bool = True
    validate_duplicates: bool = True

    # Progress tracking
    report_progress_every: int = 100
    save_failed_rows: bool = True


class BatchImporter(BaseIngestionHandler):
    """
    Handles batch import of identities from CSV and JSON files.

    Supports streaming large files, field mapping, validation,
    and progress tracking with detailed error reporting.
    """

    def __init__(self, db_session: AsyncSession, saas_app: SaaSApplication,
                 config: Optional[BatchImportConfig] = None):
        super().__init__(db_session, saas_app)
        self.config = config or BatchImportConfig()

        # Default field mappings by provider
        self.default_mappings = {
            IdentityProvider.OKTA: {
                "login": "username",
                "email": "email",
                "firstName": "first_name",
                "lastName": "last_name",
                "displayName": "display_name",
                "department": "department",
                "title": "job_title",
                "manager": "manager_email",
                "employeeNumber": "employee_id"
            },
            IdentityProvider.AZURE_AD: {
                "userPrincipalName": "username",
                "mail": "email",
                "givenName": "first_name",
                "surname": "last_name",
                "displayName": "display_name",
                "department": "department",
                "jobTitle": "job_title",
                "manager": "manager_email",
                "employeeId": "employee_id"
            },
            # Add more provider mappings as needed
        }

        # Set field mapping
        if not self.config.field_mapping:
            self.config.field_mapping = self.default_mappings.get(
                self.provider,
                {}  # Empty mapping for unknown providers
            )

        # Set required fields
        if not self.config.required_fields:
            self.config.required_fields = ["email"]

    async def ingest(self, file: Union[UploadFile, Path, str], **kwargs) -> IngestionResult:
        """
        Import identities from a file.

        Args:
            file: File to import (UploadFile, Path, or string path)
            **kwargs: Additional options

        Returns:
            IngestionResult with import statistics
        """
        self.result.status = IngestionStatus.IN_PROGRESS

        with logfire.span("Batch import", provider=self.provider):
            try:
                # Determine file type and process accordingly
                if isinstance(file, UploadFile):
                    file_type = self._get_file_type_from_upload(file)
                    await self._validate_file_size(file)
                else:
                    file_path = Path(file) if isinstance(file, str) else file
                    file_type = file_path.suffix.lower()
                    await self._validate_file_path(file_path)

                # Process based on file type
                if file_type in [".csv", "text/csv"]:
                    await self._process_csv_file(file)
                elif file_type in [".json", "application/json"]:
                    await self._process_json_file(file)
                else:
                    raise IngestionError(
                        f"Unsupported file type: {file_type}",
                        provider=self.provider,
                        error_code="UNSUPPORTED_FILE_TYPE"
                    )

                # Update SaaS app stats
                self.saas_app.record_sync_success(
                    identities_count=self.result.processed_records,
                    duration_seconds=int((datetime.utcnow() - self.result.started_at).total_seconds())
                )

                self.result.complete()
                await self.db_session.commit()

                logfire.info(
                    "Batch import completed",
                    provider=self.provider,
                    imported=self.result.created_records + self.result.updated_records,
                    failed=self.result.failed_records,
                    skipped=self.result.skipped_records
                )

            except Exception as e:
                logfire.error(
                    "Batch import failed",
                    provider=self.provider,
                    error=str(e)
                )
                self.result.status = IngestionStatus.FAILED
                self.result.add_error(str(e))
                self.saas_app.record_sync_error(str(e))
                await self.db_session.rollback()
                raise

        return self.result

    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate identity data from file."""
        # Check required fields
        for field in self.config.required_fields:
            if field not in data or not data[field]:
                return False

        # Validate email format
        if self.config.validate_email and "email" in data:
            email = data["email"]
            if not email or "@" not in email:
                return False

        return True

    async def transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform file data to internal identity format."""
        # Apply field mapping
        transformed = {}

        for source_field, target_field in self.config.field_mapping.items():
            if source_field in data:
                transformed[target_field] = data[source_field]

        # Add default values
        transformed.setdefault("external_id", transformed.get("email", "").split("@")[0])
        transformed.setdefault("username", transformed.get("email", "").split("@")[0])
        transformed.setdefault("display_name", f"{transformed.get('first_name', '')} {transformed.get('last_name', '')}".strip())

        # Add provider attributes
        transformed["provider_attributes"] = {
            "imported_at": datetime.utcnow().isoformat(),
            "import_source": "batch",
            "original_data": data
        }

        return transformed

    def _get_file_type_from_upload(self, file: UploadFile) -> str:
        """Get file type from UploadFile."""
        if file.content_type:
            return file.content_type

        # Fallback to extension
        if file.filename:
            return Path(file.filename).suffix.lower()

        return ""

    async def _validate_file_size(self, file: UploadFile) -> None:
        """Validate uploaded file size."""
        # Reset file position
        await file.seek(0)

        # Check size
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)

        if size_mb > self.config.max_file_size_mb:
            raise IngestionError(
                f"File size {size_mb:.1f}MB exceeds maximum {self.config.max_file_size_mb}MB",
                provider=self.provider,
                error_code="FILE_TOO_LARGE"
            )

        # Reset for reading
        await file.seek(0)

    async def _validate_file_path(self, file_path: Path) -> None:
        """Validate file path exists and size."""
        if not file_path.exists():
            raise IngestionError(
                f"File not found: {file_path}",
                provider=self.provider,
                error_code="FILE_NOT_FOUND"
            )

        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.max_file_size_mb:
            raise IngestionError(
                f"File size {size_mb:.1f}MB exceeds maximum {self.config.max_file_size_mb}MB",
                provider=self.provider,
                error_code="FILE_TOO_LARGE"
            )

    async def _process_csv_file(self, file: Union[UploadFile, Path]) -> None:
        """Process CSV file."""
        failed_rows = []

        try:
            # Read CSV in chunks
            async for chunk in self._read_csv_chunks(file):
                identities = []

                for row_num, row in chunk.iterrows():
                    try:
                        # Convert row to dict
                        row_data = row.to_dict()

                        # Validate
                        if not await self.validate_data(row_data):
                            if self.config.skip_invalid_rows:
                                self.result.skipped_records += 1
                                self.result.add_warning(f"Row {row_num}: Invalid data")
                                if self.config.save_failed_rows:
                                    failed_rows.append({"row": row_num, "data": row_data, "error": "Validation failed"})
                                continue
                            else:
                                raise IngestionError(f"Row {row_num}: Invalid data")

                        # Transform
                        identity_data = await self.transform_data(row_data)
                        identities.append(identity_data)

                    except Exception as e:
                        if self.config.skip_invalid_rows:
                            self.result.add_error(f"Row {row_num}: {str(e)}")
                            if self.config.save_failed_rows:
                                failed_rows.append({"row": row_num, "data": row_data, "error": str(e)})
                        else:
                            raise

                # Process batch
                if identities:
                    await self.process_batch(identities, batch_size=self.config.chunk_size)

                # Report progress
                if self.result.processed_records % self.config.report_progress_every == 0:
                    logfire.info(
                        "Batch import progress",
                        provider=self.provider,
                        processed=self.result.processed_records,
                        failed=self.result.failed_records
                    )

        finally:
            # Save failed rows if configured
            if self.config.save_failed_rows and failed_rows:
                await self._save_failed_rows(failed_rows)

    async def _process_json_file(self, file: Union[UploadFile, Path]) -> None:
        """Process JSON file."""
        failed_records = []

        try:
            # Read JSON file
            if isinstance(file, UploadFile):
                content = await file.read()
                data = json.loads(content)
            else:
                async with aiofiles.open(file, mode='r') as f:
                    content = await f.read()
                    data = json.loads(content)

            # Handle both array and object formats
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                # Look for common array fields
                records = (
                    data.get("users") or
                    data.get("identities") or
                    data.get("data") or
                    data.get("records") or
                    []
                )
            else:
                raise IngestionError("Invalid JSON format")

            self.result.total_records = len(records)

            # Process in chunks
            for i in range(0, len(records), self.config.chunk_size):
                chunk = records[i:i + self.config.chunk_size]
                identities = []

                for idx, record in enumerate(chunk):
                    record_num = i + idx

                    try:
                        # Validate
                        if not await self.validate_data(record):
                            if self.config.skip_invalid_rows:
                                self.result.skipped_records += 1
                                self.result.add_warning(f"Record {record_num}: Invalid data")
                                if self.config.save_failed_rows:
                                    failed_records.append({"index": record_num, "data": record, "error": "Validation failed"})
                                continue
                            else:
                                raise IngestionError(f"Record {record_num}: Invalid data")

                        # Transform
                        identity_data = await self.transform_data(record)
                        identities.append(identity_data)

                    except Exception as e:
                        if self.config.skip_invalid_rows:
                            self.result.add_error(f"Record {record_num}: {str(e)}")
                            if self.config.save_failed_rows:
                                failed_records.append({"index": record_num, "data": record, "error": str(e)})
                        else:
                            raise

                # Process batch
                if identities:
                    await self.process_batch(identities, batch_size=self.config.chunk_size)

                # Report progress
                if self.result.processed_records % self.config.report_progress_every == 0:
                    logfire.info(
                        "Batch import progress",
                        provider=self.provider,
                        processed=self.result.processed_records,
                        failed=self.result.failed_records
                    )

        finally:
            # Save failed records if configured
            if self.config.save_failed_rows and failed_records:
                await self._save_failed_records(failed_records)

    async def _read_csv_chunks(self, file: Union[UploadFile, Path]) -> AsyncGenerator[pd.DataFrame, None]:
        """Read CSV file in chunks."""
        # Detect encoding if not specified
        if not self.config.encoding:
            if isinstance(file, UploadFile):
                sample = await file.read(10000)
                await file.seek(0)
                detected = chardet.detect(sample)
                self.config.encoding = detected['encoding'] or 'utf-8'
            else:
                with open(file, 'rb') as f:
                    sample = f.read(10000)
                    detected = chardet.detect(sample)
                    self.config.encoding = detected['encoding'] or 'utf-8'

        # Read CSV
        if isinstance(file, UploadFile):
            # For UploadFile, read all content and use StringIO
            content = await file.read()
            text_content = content.decode(self.config.encoding)

            # Use pandas to read CSV in chunks
            for chunk in pd.read_csv(
                io.StringIO(text_content),
                chunksize=self.config.chunk_size,
                delimiter=self.config.delimiter,
                quotechar=self.config.quotechar,
                header=0 if self.config.has_header else None
            ):
                yield chunk
        else:
            # For file path, use pandas directly
            for chunk in pd.read_csv(
                file,
                chunksize=self.config.chunk_size,
                delimiter=self.config.delimiter,
                quotechar=self.config.quotechar,
                header=0 if self.config.has_header else None,
                encoding=self.config.encoding
            ):
                yield chunk

    async def _save_failed_rows(self, failed_rows: List[Dict[str, Any]]) -> None:
        """Save failed rows to a file for review."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"failed_import_{self.provider}_{timestamp}.json"

        failed_data = {
            "provider": self.provider,
            "import_time": datetime.utcnow().isoformat(),
            "total_failed": len(failed_rows),
            "failed_rows": failed_rows
        }

        # Save to file
        async with aiofiles.open(filename, mode='w') as f:
            await f.write(json.dumps(failed_data, indent=2, default=str))

        self.result.metadata["failed_rows_file"] = filename
        logfire.info(
            "Failed rows saved",
            provider=self.provider,
            filename=filename,
            count=len(failed_rows)
        )

    async def _save_failed_records(self, failed_records: List[Dict[str, Any]]) -> None:
        """Save failed JSON records to a file for review."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"failed_import_{self.provider}_{timestamp}.json"

        failed_data = {
            "provider": self.provider,
            "import_time": datetime.utcnow().isoformat(),
            "total_failed": len(failed_records),
            "failed_records": failed_records
        }

        # Save to file
        async with aiofiles.open(filename, mode='w') as f:
            await f.write(json.dumps(failed_data, indent=2, default=str))

        self.result.metadata["failed_records_file"] = filename
        logfire.info(
            "Failed records saved",
            provider=self.provider,
            filename=filename,
            count=len(failed_records)
        )
