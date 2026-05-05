from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from fastapi import HTTPException, Request


_PUBLIC_PATHS = frozenset(
    {
        "/health",
        "/v1/health",
        "/livez",
        "/v1/livez",
        "/readyz",
        "/v1/readyz",
        "/metrics",
        "/v1/metrics",
        "/metrics/prometheus",
        "/v1/metrics/prometheus",
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
    }
)
_MUTATION_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _bearer_token(headers: Any) -> str:
    auth_header = ""
    try:
        auth_header = str(headers.get("Authorization", "") or "").strip()
    except Exception:
        auth_header = ""
    if not auth_header:
        return ""
    prefix, _, value = auth_header.partition(" ")
    if prefix.strip().lower() != "bearer":
        return ""
    return value.strip()


def _api_key(headers: Any) -> str:
    try:
        return str(headers.get("X-API-Key", "") or "").strip()
    except Exception:
        return ""


def _required_scopes(claims: dict[str, object]) -> set[str]:
    scopes: set[str] = set()
    scope_value = claims.get("scope")
    if isinstance(scope_value, str):
        scopes.update(part.strip() for part in scope_value.split() if part.strip())
    scp_value = claims.get("scp")
    if isinstance(scp_value, str):
        scopes.update(part.strip() for part in scp_value.split() if part.strip())
    elif isinstance(scp_value, list):
        scopes.update(str(item).strip() for item in scp_value if str(item).strip())
    return scopes


@dataclass(frozen=True)
class ApiAuthSettings:
    mode: str = "off"
    scope: str = "mutations"
    legacy_token: str | None = None
    jwt_secret: str | None = None
    jwks_url: str | None = None
    jwt_algorithms: tuple[str, ...] = ("RS256",)
    jwt_issuer: str | None = None
    jwt_audience: tuple[str, ...] = ()
    jwt_required_scopes: tuple[str, ...] = ()
    jwt_leeway_seconds: int = 0

    @classmethod
    def from_values(
        cls,
        *,
        mode: str | None,
        scope: str | None,
        legacy_token: str | None,
        jwt_secret: str | None,
        jwks_url: str | None,
        jwt_algorithms: str | tuple[str, ...] | list[str] | None,
        jwt_issuer: str | None,
        jwt_audience: str | tuple[str, ...] | list[str] | None,
        jwt_required_scopes: str | tuple[str, ...] | list[str] | None,
        jwt_leeway_seconds: int | None,
    ) -> "ApiAuthSettings":
        normalized_mode = str(mode or "").strip().lower()
        normalized_scope = str(scope or "mutations").strip().lower() or "mutations"
        normalized_legacy_token = str(legacy_token or "").strip() or None
        normalized_jwt_secret = str(jwt_secret or "").strip() or None
        normalized_jwks_url = str(jwks_url or "").strip() or None

        if isinstance(jwt_algorithms, str):
            normalized_algorithms = _split_csv(jwt_algorithms) or ("RS256",)
        elif isinstance(jwt_algorithms, (list, tuple)):
            normalized_algorithms = tuple(str(item).strip() for item in jwt_algorithms if str(item).strip()) or ("RS256",)
        else:
            normalized_algorithms = ("RS256",)

        if isinstance(jwt_audience, str):
            normalized_audience = _split_csv(jwt_audience)
        elif isinstance(jwt_audience, (list, tuple)):
            normalized_audience = tuple(str(item).strip() for item in jwt_audience if str(item).strip())
        else:
            normalized_audience = ()

        if isinstance(jwt_required_scopes, str):
            normalized_required_scopes = _split_csv(jwt_required_scopes)
        elif isinstance(jwt_required_scopes, (list, tuple)):
            normalized_required_scopes = tuple(str(item).strip() for item in jwt_required_scopes if str(item).strip())
        else:
            normalized_required_scopes = ()

        if normalized_mode in {"", "auto"}:
            if normalized_jwt_secret or normalized_jwks_url:
                normalized_mode = "jwt"
            elif normalized_legacy_token:
                normalized_mode = "token"
            else:
                normalized_mode = "off"

        if normalized_mode not in {"off", "token", "jwt", "token_or_jwt"}:
            normalized_mode = "off"
        if normalized_scope not in {"mutations", "all"}:
            normalized_scope = "mutations"

        return cls(
            mode=normalized_mode,
            scope=normalized_scope,
            legacy_token=normalized_legacy_token,
            jwt_secret=normalized_jwt_secret,
            jwks_url=normalized_jwks_url,
            jwt_algorithms=normalized_algorithms,
            jwt_issuer=(str(jwt_issuer or "").strip() or None),
            jwt_audience=normalized_audience,
            jwt_required_scopes=normalized_required_scopes,
            jwt_leeway_seconds=max(0, int(jwt_leeway_seconds or 0)),
        )

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


class ApiAuthenticator:
    def __init__(self, *, settings: ApiAuthSettings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self._jwks_client = None

    def summary(self) -> dict[str, object]:
        return {
            "enabled": self.settings.enabled,
            "mode": self.settings.mode,
            "scope": self.settings.scope,
            "jwt_issuer": self.settings.jwt_issuer or "",
            "jwt_audience": list(self.settings.jwt_audience),
            "jwt_required_scopes": list(self.settings.jwt_required_scopes),
            "jwks_url_configured": bool(self.settings.jwks_url),
            "legacy_token_configured": bool(self.settings.legacy_token),
        }

    def request_requires_auth(self, *, path: str, method: str) -> bool:
        if not self.settings.enabled:
            return False
        if path in _PUBLIC_PATHS or path.startswith("/docs"):
            return False
        if self.settings.scope == "all":
            return True
        return method.upper() in _MUTATION_METHODS

    def authenticate_request(self, request: Request) -> dict[str, object] | None:
        path = str(request.url.path)
        method = str(request.method or "GET").upper()
        if not self.request_requires_auth(path=path, method=method):
            return None

        principal = self._authenticate_headers(request.headers)
        request.state.auth_principal = principal
        request.state.auth_type = str(principal.get("auth_type", ""))
        request.state.auth_subject = str(principal.get("subject", ""))
        return principal

    def _authenticate_headers(self, headers: Any) -> dict[str, object]:
        token_candidate = _bearer_token(headers)
        api_key_candidate = _api_key(headers)

        if self.settings.mode in {"token", "token_or_jwt"} and self.settings.legacy_token:
            if token_candidate == self.settings.legacy_token or api_key_candidate == self.settings.legacy_token:
                return {
                    "auth_type": "token",
                    "subject": "api-token",
                    "claims": {},
                }
            if self.settings.mode == "token":
                raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Missing or invalid API token."})

        if self.settings.mode in {"jwt", "token_or_jwt"}:
            if not token_candidate:
                raise HTTPException(
                    status_code=401,
                    detail={"code": "unauthorized", "message": "Missing Bearer token for JWT authentication."},
                )
            claims = self._decode_jwt(token_candidate)
            subject = str(claims.get("sub", "") or claims.get("client_id", "") or claims.get("azp", "")).strip()
            return {
                "auth_type": "jwt",
                "subject": subject,
                "claims": claims,
            }

        raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Authentication is not configured."})

    def _decode_jwt(self, token: str) -> dict[str, object]:
        try:
            import jwt
        except Exception as exc:  # pragma: no cover - environment dependency guard
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "auth_misconfigured",
                    "message": "PyJWT is required for JWT authentication support.",
                },
            ) from exc

        algorithms = list(self.settings.jwt_algorithms or ("RS256",))
        options: dict[str, object] = {"verify_aud": bool(self.settings.jwt_audience)}
        kwargs: dict[str, object] = {
            "algorithms": algorithms,
            "options": options,
            "leeway": int(self.settings.jwt_leeway_seconds),
        }
        if self.settings.jwt_issuer:
            kwargs["issuer"] = self.settings.jwt_issuer
        if self.settings.jwt_audience:
            kwargs["audience"] = self.settings.jwt_audience[0] if len(self.settings.jwt_audience) == 1 else list(self.settings.jwt_audience)

        try:
            if self.settings.jwks_url:
                if self._jwks_client is None:
                    self._jwks_client = jwt.PyJWKClient(self.settings.jwks_url)
                jwks_client = self._jwks_client
                if jwks_client is None:  # pragma: no cover - defensive guard
                    raise HTTPException(
                        status_code=500,
                        detail={"code": "auth_misconfigured", "message": "JWKS client initialization failed."},
                    )
                signing_key = jwks_client.get_signing_key_from_jwt(token).key
                claims_obj = jwt.decode(token, signing_key, **kwargs)
            elif self.settings.jwt_secret:
                claims_obj = jwt.decode(token, self.settings.jwt_secret, **kwargs)
            else:
                raise HTTPException(
                    status_code=500,
                    detail={"code": "auth_misconfigured", "message": "JWT authentication is enabled without a key source."},
                )
        except HTTPException:
            raise
        except Exception as exc:
            self.logger.warning("JWT verification failed: %s", exc)
            raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "JWT verification failed."}) from exc

        claims = claims_obj if isinstance(claims_obj, dict) else {}
        if self.settings.jwt_required_scopes:
            present_scopes = _required_scopes(claims)
            required = {scope for scope in self.settings.jwt_required_scopes if scope}
            if not required.issubset(present_scopes):
                raise HTTPException(
                    status_code=403,
                    detail={
                        "code": "insufficient_scope",
                        "message": "JWT is missing required scopes.",
                        "details": {
                            "required_scopes": sorted(required),
                            "present_scopes": sorted(present_scopes),
                        },
                    },
                )
        return claims
