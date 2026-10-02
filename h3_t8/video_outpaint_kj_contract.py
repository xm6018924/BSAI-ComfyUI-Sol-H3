"""Exact audited KJ revisions; LF/CRLF remain distinct cache identities."""
import hashlib

KJ_SOURCE_SHA256 = "c371576b1bb31a2f518bdb4ceda43cb10b20338f0c9d68f99ed1be76ce06478f"
KJ_REVISIONS = {
    KJ_SOURCE_SHA256: ("3f20054", False),
    "6f3df10c042677270053d75223db3e05ede1106be20fc178220531b73dc868fa": ("3f20054", False),
    "acbfdd2c25ebec34b1ade23d4856931209a9e1d5b690b810f2cef0af47832642": ("da90cca", True),
    "c67d638cd1f060ff4dd53dc3843aab77b980ea783b24ca4271300cbe040cb91b": ("da90cca", True),
}


def kj_source_contract(payload):
    if not isinstance(payload, bytes):
        raise ValueError("KJ source evidence must be bytes")
    digest = hashlib.sha256(payload).hexdigest()
    if digest not in KJ_REVISIONS:
        raise ValueError("KJ memory implementation changed; this revision needs a compatibility audit")
    revision, callback = KJ_REVISIONS[digest]
    return {"source_sha256": digest, "revision": revision, "attention_callback": callback}
