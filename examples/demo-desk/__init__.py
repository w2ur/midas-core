"""Marker that keeps the packaged demo desk a *regular* package.

The desk ships in the wheel as ``midas_demo_desk`` and ``midas init-demo`` reads
it back through ``importlib.resources``. Without this file the name resolves as a
namespace package, and an editable install contributes a second, non-directory
portion (``__editable__.midas_core-<v>.finder.__path_hook__``) — enough for
``files()`` to raise ``NotADirectoryError``. A regular package resolves to
exactly one directory, whatever else is on the path.

``midas init-demo`` never copies this file into a user's desk.
"""
