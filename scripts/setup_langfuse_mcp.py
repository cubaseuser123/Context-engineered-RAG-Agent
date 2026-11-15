"""Register and verify the official project-scoped Langfuse MCP, without printing secrets."""
import base64
import json
import os
from pathlib import Path
import tomllib

import requests
from dotenv import load_dotenv


def main():
    load_dotenv()
    endpoint = os.environ["LANGFUSE_BASE_URL"].rstrip("/") + "/api/public/mcp"
    token = base64.b64encode((os.environ["LANGFUSE_PUBLIC_KEY"] + ":" + os.environ["LANGFUSE_SECRET_KEY"]).encode()).decode()
    config = Path.home() / ".codex" / "config.toml"
    original = config.read_text(encoding="utf-8")
    parsed = tomllib.loads(original)
    if "langfuse" not in parsed.get("mcp_servers", {}):
        config.write_text(original.rstrip() + "\n\n[mcp_servers.langfuse]\nurl = " + json.dumps(endpoint) + '\nhttp_headers = { "Authorization" = ' + json.dumps("Basic " + token) + ' }\n', encoding="utf-8")
    else:
        raise RuntimeError("Langfuse configuration already exists; inspect before changing it")
    print("Registered official Langfuse MCP in Codex configuration.")
    headers = {"Authorization": "Basic " + token, "Accept": "application/json, text/event-stream"}
    session = requests.Session()
    def rpc(method, params, request_id):
        response = session.post(endpoint, headers=headers, json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}, timeout=30)
        response.raise_for_status()
        if response.headers.get("mcp-session-id"):
            headers["mcp-session-id"] = response.headers["mcp-session-id"]
        try:
            return response.json()
        except ValueError:
            return json.loads(next(line[6:] for line in response.text.splitlines() if line.startswith("data: ")))
    initialized = rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "codex-langfuse-setup", "version": "1.0"}}, 1)
    if "error" in initialized:
        raise RuntimeError(initialized["error"])
    tools = rpc("tools/list", {}, 2)
    if "error" in tools:
        raise RuntimeError(tools["error"])
    listing = tools["result"]["tools"]
    Path(".cache/langfuse-mcp-tools.json").write_text(json.dumps(listing, indent=2), encoding="utf-8")
    print("Authenticated MCP handshake successful. Tools:", ", ".join(t["name"] for t in listing))


if __name__ == "__main__":
    main()
