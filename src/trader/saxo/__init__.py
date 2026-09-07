"""Saxo Bank OpenAPI client layer.

Intentionally free of re-exports. ``trader.config`` imports
``trader.saxo.environments`` for the ``Environment`` enum, and eagerly importing
``auth`` or ``client`` here -- both of which import ``trader.config`` -- would
close that loop into a circular import. Import submodules directly:

    from trader.saxo.client import SaxoClient
"""
