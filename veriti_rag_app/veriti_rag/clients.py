"""Persistent client registry and client-specific workspace paths."""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
import json
import re
import unicodedata
from pathlib import Path

from .config import Settings

DEFAULT_CLIENT_ID = "maven_demo"
CLIENT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def registry_path(s: Settings) -> Path:
    return Path(s.duckdb_path).parent / "clients.json"


def _read_registry(s: Settings) -> dict:
    path = registry_path(s)
    if not path.exists():
        return {"clients": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        clients = data.get("clients", {})
        if not isinstance(clients, dict):
            raise ValueError("clients must be an object")
        return {"clients": clients}
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise ValueError(f"Client registry is unreadable: {path}") from exc


def _write_registry(s: Settings, data: dict) -> None:
    path = registry_path(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def list_clients(s: Settings) -> list[dict[str, str]]:
    """Return registered clients, registering the legacy default on first use."""
    data = _read_registry(s)
    clients = data["clients"]
    if DEFAULT_CLIENT_ID not in clients:
        clients[DEFAULT_CLIENT_ID] = {
            "name": "Maven Demo",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_registry(s, data)
    discovered = set()
    for root in (Path(s.kb_dir), Path(s.data_dir)):
        if root.exists():
            discovered.update(
                child.name for child in root.iterdir()
                if child.is_dir() and CLIENT_ID_RE.fullmatch(child.name)
            )
    changed = False
    for client_id in discovered - clients.keys():
        clients[client_id] = {
            "name": client_id.replace("_", " ").replace("-", " ").title(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        changed = True
    if changed:
        _write_registry(s, data)
    return [
        {"client_id": client_id, **details}
        for client_id, details in sorted(clients.items(), key=lambda item: item[1].get("name", item[0]).casefold())
        if CLIENT_ID_RE.fullmatch(client_id)
    ]


def client_name(s: Settings, client_id: str) -> str:
    for client in list_clients(s):
        if client["client_id"] == client_id:
            return client["name"]
    raise KeyError(f"Unknown client: {client_id}")


def _slug(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.strip()).strip("_").lower()
    return value[:64].rstrip("_-")


def create_client(s: Settings, name: str) -> dict[str, str]:
    """Create a registered client and its empty KB, CSV, and index folders."""
    display_name = " ".join(name.split())
    if not display_name or len(display_name) > 100 or any(ord(ch) < 32 for ch in display_name):
        raise ValueError("Enter a client name between 1 and 100 printable characters.")

    data = _read_registry(s)
    clients = data["clients"]
    if any(info.get("name", "").casefold() == display_name.casefold() for info in clients.values()):
        raise ValueError(f"A client named '{display_name}' is already registered.")

    base_id = _slug(display_name)
    if not base_id:
        raise ValueError("Client name must contain at least one English letter or number.")

    kb_root = Path(s.kb_dir)
    data_root = Path(s.data_dir)
    index_root = Path(s.index_dir)
    client_id = base_id
    suffix = 2
    while (client_id in clients or (kb_root / client_id).exists()
           or (data_root / client_id).exists() or (index_root / client_id).exists()):
        suffix_text = f"_{suffix}"
        client_id = f"{base_id[:64 - len(suffix_text)]}{suffix_text}"
        suffix += 1
    if not CLIENT_ID_RE.fullmatch(client_id):
        raise ValueError("Could not generate a safe client ID from that name.")

    paths = [kb_root / client_id, data_root / client_id, index_root / client_id]
    created = []
    try:
        for path in paths:
            path.mkdir(parents=True, exist_ok=False)
            created.append(path)
        (paths[0] / "_meta.yml").write_text("version: v1\n", encoding="utf-8")
        clients[client_id] = {
            "name": display_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_registry(s, data)
    except Exception:
        # Roll back only directories created by this call; never touch pre-existing paths.
        for path in reversed(created):
            try:
                for child in path.iterdir():
                    child.unlink()
                path.rmdir()
            except OSError:
                pass
        raise

    return {"client_id": client_id, **clients[client_id]}


def client_data_dir(s: Settings, client_id: str) -> Path:
    """Resolve the client's CSV folder, retaining the legacy maven_demo location."""
    if not CLIENT_ID_RE.fullmatch(client_id):
        raise ValueError("Invalid client ID.")
    root = Path(s.data_dir)
    per_client = root / client_id
    if client_id == DEFAULT_CLIENT_ID and not per_client.exists() and any(root.glob("*.csv")):
        return root
    per_client.mkdir(parents=True, exist_ok=True)
    return per_client


def settings_for_client(s: Settings, client_id: str) -> Settings:
    """Return settings with explicit selected-client paths."""
    client_name(s, client_id)  # Require an entry in the persistent registry.
    data_dir = client_data_dir(s, client_id)
    index_dir = Path(s.index_dir) if client_id == DEFAULT_CLIENT_ID else Path(s.index_dir) / client_id
    index_dir.mkdir(parents=True, exist_ok=True)
    return replace(s, client_id=client_id, data_dir=str(data_dir), index_dir=str(index_dir))
