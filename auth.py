"""
auth.py — Daily rotating password system
-----------------------------------------
SECURITY FIX: All string comparisons now use hmac.compare_digest()
which runs in constant time and prevents timing attacks.
(Plain == comparison leaks information about how many characters match.)
"""
import os
import hashlib
import hmac
from datetime import date
from dotenv import load_dotenv

load_dotenv()


def get_daily_password() -> str:
    """
    Derives today's guest code from the master admin key + today's date.
    The code changes every midnight automatically.
    Nobody can predict tomorrow's code without the master key.
    """
    master = os.getenv("MASTER_ADMIN_KEY", "")
    if not master:
        raise RuntimeError("MASTER_ADMIN_KEY is not set in .env")
    today = date.today().isoformat()   # e.g. "2025-03-02"
    raw = hashlib.sha256(f"{master}:{today}".encode()).hexdigest()
    # Return first 8 chars — short enough to type, long enough to be unguessable
    return raw[:8]


def verify_password(candidate: str) -> bool:
    """
    FIX: use hmac.compare_digest for constant-time comparison.
    Plain == leaks timing info (returns faster when more chars match),
    which could theoretically be exploited to brute-force the code
    character-by-character. compare_digest always takes the same time.
    """
    if not candidate:
        return False
    expected = get_daily_password()
    # Both must be the same type (str); encode to bytes for compare_digest
    return hmac.compare_digest(
        expected.encode("utf-8"),
        candidate.strip().encode("utf-8"),
    )


def verify_admin_key(candidate: str) -> bool:
    """
    FIX: same constant-time comparison for the master admin key itself.
    """
    if not candidate:
        return False
    master = os.getenv("MASTER_ADMIN_KEY", "")
    return hmac.compare_digest(
        master.encode("utf-8"),
        candidate.strip().encode("utf-8"),
    )
