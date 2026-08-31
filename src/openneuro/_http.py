"""HTTP constants shared by the metadata and download paths.

These live in their own module so that `_download` and `_nemar` can both use
them without importing one another.

niquests verifies TLS certificates against the operating system's native trust
store by default (via wassima), which covers users in enterprise environments
with custom CAs. No explicit SSL context is required, and the same verification
applies across the sync and async sessions used throughout the package.
"""

import niquests

from openneuro import __version__

user_agent_header: dict[str, str] = {"user-agent": f"openneuro-py/{__version__}"}

# HTTP server responses that indicate hopefully intermittent errors that
# warrant a retry.
allowed_retry_codes = (408, 500, 502, 503, 504, 522, 524)

allowed_retry_exceptions = (
    # Connection errors (refused/reset/aborted) and DNS failures
    # ("[Errno -3] Temporary failure in name resolution"). Connect timeouts land
    # here too, since ``ConnectTimeout`` is itself a ``ConnectionError``.
    niquests.ConnectionError,
    # Read timeouts are a ``Timeout``, not a ``ConnectionError``, so list them
    # separately.
    niquests.ReadTimeout,
    # Incomplete chunked reads: "peer closed connection without sending
    # complete message body".
    niquests.exceptions.ChunkedEncodingError,
)
