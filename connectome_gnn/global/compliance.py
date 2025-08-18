"""Compliance management for GDPR, CCPA, and other privacy regulations."""

import json
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

from ..robust.logging_config import get_logger
from ..robust.security import SecurityManager


class DataProcessingLawfulBasis(Enum):
    """GDPR lawful basis for processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class DataSubjectRights(Enum):
    """Data subject rights under privacy regulations."""
    ACCESS = "access"  # Right to access personal data
    RECTIFICATION = "rectification"  # Right to correct data
    ERASURE = "erasure"  # Right to be forgotten
    RESTRICT_PROCESSING = "restrict_processing"  # Right to restrict processing
    DATA_PORTABILITY = "data_portability"  # Right to data portability
    OBJECT = "object"  # Right to object to processing
    WITHDRAW_CONSENT = "withdraw_consent"  # Right to withdraw consent


@dataclass
class DataSubject:
    """Represents a data subject under privacy regulations."""
    subject_id: str
    email: Optional[str] = None
    jurisdiction: Optional[str] = None
    consent_given: bool = False
    consent_date: Optional[datetime] = None
    consent_version: Optional[str] = None
    lawful_basis: Optional[DataProcessingLawfulBasis] = None
    data_categories: Set[str] = None
    retention_period: Optional[int] = None  # Days
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.data_categories is None:
            self.data_categories = set()
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class DataProcessingActivity:
    """Records data processing activities."""
    activity_id: str
    subject_id: str
    activity_type: str
    data_categories: Set[str]
    purpose: str
    lawful_basis: DataProcessingLawfulBasis
    processing_date: datetime
    retention_period: Optional[int] = None
    third_parties: Optional[List[str]] = None
    cross_border_transfer: bool = False
    transfer_countries: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.third_parties is None:
            self.third_parties = []
        if self.transfer_countries is None:
            self.transfer_countries = []


class ComplianceManager:
    """Base compliance manager for privacy regulations."""
    
    def __init__(self, jurisdiction: str = "EU", 
                 data_dir: str = "./compliance_data",
                 logger: Optional[logging.Logger] = None):
        self.jurisdiction = jurisdiction
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or get_logger("compliance_manager")
        self.security_manager = SecurityManager()
        
        # Storage
        self.data_subjects: Dict[str, DataSubject] = {}
        self.processing_activities: List[DataProcessingActivity] = []
        self.consent_records: Dict[str, Dict] = {}
        
        # Load existing data
        self._load_compliance_data()
    
    def _load_compliance_data(self):
        """Load compliance data from storage."""
        subjects_file = self.data_dir / "data_subjects.json"
        activities_file = self.data_dir / "processing_activities.json"
        consent_file = self.data_dir / "consent_records.json"
        
        # Load data subjects
        if subjects_file.exists():
            try:
                with open(subjects_file, 'r') as f:
                    subjects_data = json.load(f)
                    for subject_data in subjects_data:
                        # Convert datetime strings back to datetime objects
                        if subject_data.get('consent_date'):
                            subject_data['consent_date'] = datetime.fromisoformat(subject_data['consent_date'])
                        if subject_data.get('created_at'):
                            subject_data['created_at'] = datetime.fromisoformat(subject_data['created_at'])
                        if subject_data.get('updated_at'):
                            subject_data['updated_at'] = datetime.fromisoformat(subject_data['updated_at'])
                        if subject_data.get('data_categories'):
                            subject_data['data_categories'] = set(subject_data['data_categories'])
                        if subject_data.get('lawful_basis'):
                            subject_data['lawful_basis'] = DataProcessingLawfulBasis(subject_data['lawful_basis'])
                        
                        subject = DataSubject(**subject_data)
                        self.data_subjects[subject.subject_id] = subject
            except Exception as e:
                self.logger.error(f"Failed to load data subjects: {e}")
        
        # Load processing activities
        if activities_file.exists():
            try:
                with open(activities_file, 'r') as f:
                    activities_data = json.load(f)
                    for activity_data in activities_data:
                        if activity_data.get('processing_date'):
                            activity_data['processing_date'] = datetime.fromisoformat(activity_data['processing_date'])
                        if activity_data.get('data_categories'):
                            activity_data['data_categories'] = set(activity_data['data_categories'])
                        if activity_data.get('lawful_basis'):
                            activity_data['lawful_basis'] = DataProcessingLawfulBasis(activity_data['lawful_basis'])
                        
                        activity = DataProcessingActivity(**activity_data)
                        self.processing_activities.append(activity)
            except Exception as e:
                self.logger.error(f"Failed to load processing activities: {e}")
        
        # Load consent records
        if consent_file.exists():
            try:
                with open(consent_file, 'r') as f:
                    self.consent_records = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load consent records: {e}")
    
    def _save_compliance_data(self):
        """Save compliance data to storage."""
        # Save data subjects
        subjects_file = self.data_dir / "data_subjects.json"
        subjects_data = []
        for subject in self.data_subjects.values():
            subject_dict = asdict(subject)
            # Convert datetime objects to ISO strings
            if subject_dict.get('consent_date'):
                subject_dict['consent_date'] = subject_dict['consent_date'].isoformat()
            if subject_dict.get('created_at'):
                subject_dict['created_at'] = subject_dict['created_at'].isoformat()
            if subject_dict.get('updated_at'):
                subject_dict['updated_at'] = subject_dict['updated_at'].isoformat()
            if subject_dict.get('data_categories'):
                subject_dict['data_categories'] = list(subject_dict['data_categories'])
            if subject_dict.get('lawful_basis'):
                subject_dict['lawful_basis'] = subject_dict['lawful_basis'].value
            subjects_data.append(subject_dict)
        
        try:
            with open(subjects_file, 'w') as f:
                json.dump(subjects_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save data subjects: {e}")
        
        # Save processing activities
        activities_file = self.data_dir / "processing_activities.json"
        activities_data = []
        for activity in self.processing_activities:
            activity_dict = asdict(activity)
            if activity_dict.get('processing_date'):
                activity_dict['processing_date'] = activity_dict['processing_date'].isoformat()
            if activity_dict.get('data_categories'):
                activity_dict['data_categories'] = list(activity_dict['data_categories'])
            if activity_dict.get('lawful_basis'):
                activity_dict['lawful_basis'] = activity_dict['lawful_basis'].value
            activities_data.append(activity_dict)
        
        try:
            with open(activities_file, 'w') as f:
                json.dump(activities_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save processing activities: {e}")
        
        # Save consent records
        consent_file = self.data_dir / "consent_records.json"
        try:
            with open(consent_file, 'w') as f:
                json.dump(self.consent_records, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save consent records: {e}")
    
    def register_data_subject(self, subject_id: str, email: str = None,
                            jurisdiction: str = None) -> DataSubject:
        """Register a new data subject."""
        subject = DataSubject(
            subject_id=subject_id,
            email=email,
            jurisdiction=jurisdiction or self.jurisdiction
        )
        
        self.data_subjects[subject_id] = subject
        self._save_compliance_data()
        
        self.logger.info(f"Registered data subject: {subject_id}")
        return subject
    
    def record_consent(self, subject_id: str, consent_given: bool,
                      consent_version: str = "1.0", purpose: str = None,
                      data_categories: List[str] = None) -> bool:
        """Record consent for data processing."""
        if subject_id not in self.data_subjects:
            self.logger.error(f"Data subject {subject_id} not found")
            return False
        
        subject = self.data_subjects[subject_id]
        subject.consent_given = consent_given
        subject.consent_date = datetime.now()
        subject.consent_version = consent_version
        subject.updated_at = datetime.now()
        
        if data_categories:
            subject.data_categories.update(data_categories)
        
        # Record consent details
        consent_record = {
            'subject_id': subject_id,
            'consent_given': consent_given,
            'consent_date': datetime.now().isoformat(),
            'consent_version': consent_version,
            'purpose': purpose,
            'data_categories': data_categories or [],
            'ip_address': None,  # Would be captured from request
            'user_agent': None   # Would be captured from request
        }
        
        consent_key = f"{subject_id}_{datetime.now().isoformat()}"
        self.consent_records[consent_key] = consent_record
        
        self._save_compliance_data()
        
        self.logger.info(f"Recorded consent for {subject_id}: {consent_given}")
        return True
    
    def record_processing_activity(self, subject_id: str, activity_type: str,
                                 data_categories: List[str], purpose: str,
                                 lawful_basis: DataProcessingLawfulBasis,
                                 retention_period: int = None) -> str:
        """Record a data processing activity."""
        activity_id = str(uuid.uuid4())
        
        activity = DataProcessingActivity(
            activity_id=activity_id,
            subject_id=subject_id,
            activity_type=activity_type,
            data_categories=set(data_categories),
            purpose=purpose,
            lawful_basis=lawful_basis,
            processing_date=datetime.now(),
            retention_period=retention_period
        )
        
        self.processing_activities.append(activity)
        self._save_compliance_data()
        
        self.logger.info(f"Recorded processing activity: {activity_id}")
        return activity_id
    
    def handle_data_subject_request(self, subject_id: str, 
                                   request_type: DataSubjectRights) -> Dict[str, Any]:
        """Handle data subject rights requests."""
        if subject_id not in self.data_subjects:
            return {
                'success': False,
                'error': f'Data subject {subject_id} not found'
            }
        
        subject = self.data_subjects[subject_id]
        
        if request_type == DataSubjectRights.ACCESS:
            return self._handle_access_request(subject_id)
        elif request_type == DataSubjectRights.ERASURE:
            return self._handle_erasure_request(subject_id)
        elif request_type == DataSubjectRights.RECTIFICATION:
            return self._handle_rectification_request(subject_id)
        elif request_type == DataSubjectRights.DATA_PORTABILITY:
            return self._handle_portability_request(subject_id)
        elif request_type == DataSubjectRights.WITHDRAW_CONSENT:
            return self._handle_consent_withdrawal(subject_id)
        else:
            return {
                'success': False,
                'error': f'Request type {request_type.value} not implemented'
            }
    
    def _handle_access_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle right to access request."""
        subject = self.data_subjects[subject_id]
        
        # Get all processing activities for this subject
        activities = [
            activity for activity in self.processing_activities
            if activity.subject_id == subject_id
        ]
        
        # Get consent records
        consent_records = {
            k: v for k, v in self.consent_records.items()
            if v['subject_id'] == subject_id
        }
        
        return {
            'success': True,
            'data': {
                'subject_info': asdict(subject),
                'processing_activities': [asdict(activity) for activity in activities],
                'consent_records': consent_records
            }
        }
    
    def _handle_erasure_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle right to be forgotten request."""
        # Check if erasure is legally possible
        subject = self.data_subjects[subject_id]
        
        # Find activities that prevent erasure
        blocking_activities = []
        for activity in self.processing_activities:
            if (activity.subject_id == subject_id and 
                activity.lawful_basis in [DataProcessingLawfulBasis.LEGAL_OBLIGATION,
                                        DataProcessingLawfulBasis.PUBLIC_TASK]):
                blocking_activities.append(activity.activity_id)
        
        if blocking_activities:
            return {
                'success': False,
                'error': 'Erasure not possible due to legal obligations',
                'blocking_activities': blocking_activities
            }
        
        # Perform erasure
        # Remove from data subjects
        del self.data_subjects[subject_id]
        
        # Remove processing activities
        self.processing_activities = [
            activity for activity in self.processing_activities
            if activity.subject_id != subject_id
        ]
        
        # Remove consent records
        self.consent_records = {
            k: v for k, v in self.consent_records.items()
            if v['subject_id'] != subject_id
        }
        
        self._save_compliance_data()
        
        self.logger.info(f"Performed data erasure for subject: {subject_id}")
        
        return {
            'success': True,
            'message': f'All data for subject {subject_id} has been erased'
        }
    
    def _handle_rectification_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle right to rectification request."""
        return {
            'success': True,
            'message': 'Rectification request noted. Please provide corrected data.',
            'instructions': 'Submit the corrected information through the appropriate channels.'
        }
    
    def _handle_portability_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle right to data portability request."""
        # Get all data for the subject in structured format
        access_data = self._handle_access_request(subject_id)
        
        if access_data['success']:
            # Format data for portability (JSON format)
            portable_data = {
                'export_date': datetime.now().isoformat(),
                'subject_id': subject_id,
                'data': access_data['data']
            }
            
            return {
                'success': True,
                'portable_data': portable_data,
                'format': 'JSON'
            }
        else:
            return access_data
    
    def _handle_consent_withdrawal(self, subject_id: str) -> Dict[str, Any]:
        """Handle consent withdrawal."""
        if subject_id in self.data_subjects:
            self.record_consent(subject_id, False, purpose="Consent withdrawal")
            
            return {
                'success': True,
                'message': f'Consent withdrawn for subject {subject_id}'
            }
        else:
            return {
                'success': False,
                'error': f'Subject {subject_id} not found'
            }
    
    def check_retention_compliance(self) -> List[str]:
        """Check for data that exceeds retention periods."""
        expired_subjects = []
        
        for subject_id, subject in self.data_subjects.items():
            if subject.retention_period:
                # Check if retention period has passed
                retention_end = subject.created_at + timedelta(days=subject.retention_period)
                if datetime.now() > retention_end:
                    expired_subjects.append(subject_id)
        
        return expired_subjects
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'jurisdiction': self.jurisdiction,
            'total_subjects': len(self.data_subjects),
            'total_activities': len(self.processing_activities),
            'consent_given_count': sum(1 for s in self.data_subjects.values() if s.consent_given),
            'lawful_basis_breakdown': {},
            'data_categories': set(),
            'expired_retentions': self.check_retention_compliance()
        }
        
        # Analyze lawful basis
        for activity in self.processing_activities:
            basis = activity.lawful_basis.value
            report['lawful_basis_breakdown'][basis] = report['lawful_basis_breakdown'].get(basis, 0) + 1
            report['data_categories'].update(activity.data_categories)
        
        report['data_categories'] = list(report['data_categories'])
        
        return report


class GDPRCompliance(ComplianceManager):
    """GDPR-specific compliance manager."""
    
    def __init__(self, **kwargs):
        super().__init__(jurisdiction="EU", **kwargs)
        self.regulation_name = "GDPR"
        self.logger.info("Initialized GDPR compliance manager")
    
    def validate_lawful_basis(self, processing_activity: DataProcessingActivity) -> bool:
        """Validate GDPR lawful basis for processing."""
        # GDPR Article 6 requires at least one lawful basis
        if not processing_activity.lawful_basis:
            return False
        
        # Additional GDPR-specific validations
        if (processing_activity.lawful_basis == DataProcessingLawfulBasis.CONSENT and
            processing_activity.subject_id in self.data_subjects):
            subject = self.data_subjects[processing_activity.subject_id]
            return subject.consent_given
        
        return True
    
    def check_cross_border_transfer_compliance(self, activity: DataProcessingActivity) -> bool:
        """Check GDPR compliance for cross-border data transfers."""
        if not activity.cross_border_transfer:
            return True
        
        # GDPR Chapter V - Transfers to third countries
        eu_countries = ["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", 
                       "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", 
                       "PL", "PT", "RO", "SK", "SI", "ES", "SE"]
        
        for country in activity.transfer_countries:
            if country not in eu_countries:
                # Would need adequacy decision or appropriate safeguards
                self.logger.warning(f"Cross-border transfer to {country} requires adequacy decision or safeguards")
                return False
        
        return True


class CCPACompliance(ComplianceManager):
    """CCPA-specific compliance manager."""
    
    def __init__(self, **kwargs):
        super().__init__(jurisdiction="CA", **kwargs)
        self.regulation_name = "CCPA"
        self.logger.info("Initialized CCPA compliance manager")
    
    def record_sale_or_sharing(self, subject_id: str, third_party: str,
                             data_categories: List[str], purpose: str) -> str:
        """Record sale or sharing of personal information."""
        return self.record_processing_activity(
            subject_id=subject_id,
            activity_type="sale_or_sharing",
            data_categories=data_categories,
            purpose=f"Sale/sharing to {third_party}: {purpose}",
            lawful_basis=DataProcessingLawfulBasis.LEGITIMATE_INTERESTS
        )
    
    def handle_opt_out_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle CCPA opt-out of sale request."""
        if subject_id not in self.data_subjects:
            return {
                'success': False,
                'error': f'Subject {subject_id} not found'
            }
        
        # Record opt-out
        subject = self.data_subjects[subject_id]
        subject.updated_at = datetime.now()
        
        # Add opt-out flag to data categories
        subject.data_categories.add("ccpa_opt_out")
        
        self._save_compliance_data()
        
        return {
            'success': True,
            'message': f'Opt-out recorded for subject {subject_id}'
        }


# Global compliance managers
_global_compliance_managers = {}

def get_compliance_manager(regulation: str = "GDPR") -> ComplianceManager:
    """Get compliance manager for specific regulation."""
    global _global_compliance_managers
    
    if regulation not in _global_compliance_managers:
        if regulation.upper() == "GDPR":
            _global_compliance_managers[regulation] = GDPRCompliance()
        elif regulation.upper() == "CCPA":
            _global_compliance_managers[regulation] = CCPACompliance()
        else:
            _global_compliance_managers[regulation] = ComplianceManager(jurisdiction=regulation)
    
    return _global_compliance_managers[regulation]