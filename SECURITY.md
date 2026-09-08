# Security policy

## Supported versions

Security fixes are applied to the latest revision of the default branch. This project does not currently maintain separate release branches.

## Reporting a vulnerability

Please do not disclose suspected vulnerabilities in a public issue. Report them privately through GitHub's **Security** tab by opening a private vulnerability report.

Include the affected component, reproduction steps, potential impact, and any suggested remediation. The project aims to acknowledge reports within seven days.

## Deployment considerations

This repository is an experimental runtime and is not a hardened multi-tenant service. Before deploying it outside a trusted environment:

- Generate unique high-entropy values for `JWT_SECRET` and `GOOGLE_TOKEN_ENCRYPTION_KEY`.
- Terminate TLS before the application and set `AUTH_COOKIE_SECURE=true`.
- Use an exact `CORS_ORIGINS` allowlist.
- Keep PostgreSQL and Qdrant private and use managed credentials, encryption, and backups.
- Accept forwarded client addresses only from trusted proxies that overwrite forwarding headers.
- Keep the reverse-proxy body limit equal to or lower than `MAX_REQUEST_BODY_BYTES`.
- Replace process-local rate limiting and session coordination before running multiple workers.
- Review Google OAuth scopes and verification requirements before enabling Drive access publicly.

Never commit environment files, API keys, OAuth credentials, encryption keys, access tokens, or refresh tokens.
