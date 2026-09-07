"""Command-line entry point.

Phase 0 exposes just enough to prove the connection works:

    trader auth login      # browser login, stores refreshable tokens
    trader auth status     # what is stored and how long it has left
    trader auth logout     # discard stored tokens
    trader account         # user, accounts, and balance from Saxo
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from trader.config import Settings, get_settings
from trader.saxo.accounts import get_balance, get_user, list_accounts
from trader.saxo.auth import AuthError, ReauthRequired, SaxoAuth
from trader.saxo.client import SaxoAPIError, SaxoClient
from trader.saxo.tokens import TokenStore


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def _cmd_auth_login(settings: Settings) -> int:
    auth = SaxoAuth(settings)
    try:
        tokens = await auth.login_interactive()
    finally:
        await auth.aclose()
    print(
        f"\nSigned in to {settings.saxo.environment.value}. "
        f"Access token valid until {tokens.access_expires_at:%H:%M:%S UTC}."
    )
    if tokens.refresh_expires_at:
        print(f"Refresh token valid until {tokens.refresh_expires_at:%H:%M:%S UTC}.")
    return 0


def _cmd_auth_status(settings: Settings) -> int:
    if settings.saxo.token_24h.get_secret_value():
        print("Using a developer-portal 24-hour token from TRADER_SAXO__TOKEN_24H (SIM only).")
        return 0

    tokens = TokenStore(settings.token_file, settings.token_key_file).load()
    if tokens is None:
        print(f"No stored tokens for {settings.saxo.environment.value}. Run: trader auth login")
        return 1

    print(f"Environment:   {settings.saxo.environment.value}")
    print(
        f"Access token:  {'EXPIRED' if tokens.access_expired() else 'valid'} "
        f"(until {tokens.access_expires_at:%Y-%m-%d %H:%M:%S UTC})"
    )
    if tokens.is_static:
        print("Refresh token: n/a (static token)")
    else:
        state = "EXPIRED" if tokens.refresh_expired() else "valid"
        until = (
            f" (until {tokens.refresh_expires_at:%Y-%m-%d %H:%M:%S UTC})"
            if tokens.refresh_expires_at
            else ""
        )
        print(f"Refresh token: {state}{until}")
    return 0


def _cmd_auth_logout(settings: Settings) -> int:
    TokenStore(settings.token_file, settings.token_key_file).clear()
    print(f"Cleared stored tokens for {settings.saxo.environment.value}.")
    return 0


async def _cmd_account(settings: Settings) -> int:
    async with SaxoClient(settings) as client:
        user = await get_user(client)
        accounts = await list_accounts(client)

        print(f"User:        {user.name or user.user_id} (UserId {user.user_id})")
        print(f"ClientKey:   {user.client_key}")
        print(f"Environment: {settings.saxo.environment.value}")
        print()

        if not accounts:
            print("No accounts returned.")
            return 1

        for account in accounts:
            balance = await get_balance(
                client, client_key=user.client_key, account_key=account.account_key
            )
            flag = "" if account.active else "  [inactive]"
            print(f"Account {account.account_id} ({account.account_type}){flag}")
            print(f"  AccountKey:       {account.account_key}")
            print(f"  Cash balance:     {balance.cash_balance:,.2f} {balance.currency}")
            print(f"  Total value:      {balance.total_value:,.2f} {balance.currency}")
            print(f"  Margin available: {balance.margin_available:,.2f} {balance.currency}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trader", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    auth = sub.add_parser("auth", help="manage Saxo authentication")
    auth_sub = auth.add_subparsers(dest="auth_command", required=True)
    auth_sub.add_parser("login", help="browser login and store tokens")
    auth_sub.add_parser("status", help="show stored token state")
    auth_sub.add_parser("logout", help="discard stored tokens")

    sub.add_parser("account", help="show user, accounts, and balances")

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    settings = get_settings()

    try:
        if args.command == "auth":
            if args.auth_command == "login":
                return asyncio.run(_cmd_auth_login(settings))
            if args.auth_command == "status":
                return _cmd_auth_status(settings)
            if args.auth_command == "logout":
                return _cmd_auth_logout(settings)
        if args.command == "account":
            return asyncio.run(_cmd_account(settings))
    except ReauthRequired as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (AuthError, SaxoAPIError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130

    parser.error(f"unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
