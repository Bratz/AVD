from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import logging
from config.settings import Settings
from services.audit.file_logger import FileLogger
from utils.exceptions import BankingJudgeError

security = HTTPBearer()
settings = Settings()
logger = logging.getLogger(__name__)
audit_logger = FileLogger()

def verify_jwt(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify JWT token from Authorization header."""
    try:
        token = credentials.credentials
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=["HS256"]
        )
        logger.debug(f"JWT verified for user: {payload.get('sub', 'unknown')}")
        audit_logger.log_event(
            event_type="jwt_verification_success",
            details={"user_id": payload.get("sub", "unknown")}
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        audit_logger.log_event(
            event_type="jwt_verification_failed",
            details={"error": "Token expired"}
        )
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {str(e)}")
        audit_logger.log_event(
            event_type="jwt_verification_failed",
            details={"error": str(e)}
        )
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"JWT verification error: {str(e)}")
        audit_logger.log_event(
            event_type="jwt_verification_error",
            details={"error": str(e)}
        )
        raise BankingJudgeError(f"JWT verification failed: {str(e)}")