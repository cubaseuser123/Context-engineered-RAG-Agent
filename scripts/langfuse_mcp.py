"""Small client for the official hosted MCP, also usable before Codex reloads tools."""
import base64
import json
import os
import requests
from dotenv import load_dotenv


class LangfuseMCP:
    def __init__(self):
        load_dotenv()
        self.url = os.environ["LANGFUSE_BASE_URL"].rstrip("/") + "/api/public/mcp"
        token = base64.b64encode((os.environ["LANGFUSE_PUBLIC_KEY"] + ":" + os.environ["LANGFUSE_SECRET_KEY"]).encode()).decode()
        self.headers = {"Authorization": "Basic " + token, "Accept": "application/json, text/event-stream"}
        self.session = requests.Session()
        self.counter = 0
        self.rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "codex-project-review", "version": "1.0"}})

    def rpc(self, method, params):
        self.counter += 1
        r = self.session.post(self.url, headers=self.headers, json={"jsonrpc": "2.0", "id": self.counter, "method": method, "params": params}, timeout=60)
        r.raise_for_status()
        if r.headers.get("mcp-session-id"):
            self.headers["mcp-session-id"] = r.headers["mcp-session-id"]
        try:
            data = r.json()
        except ValueError:
            data = json.loads(next(line[6:] for line in r.text.splitlines() if line.startswith("data: ")))
        if "error" in data:
            raise RuntimeError(data["error"])
        return data["result"]

    def call(self, name, arguments):
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise RuntimeError(result)
        if "structuredContent" in result:
            return result["structuredContent"]
        return json.loads(next(x["text"] for x in result["content"] if x["type"] == "text"))


if __name__ == "__main__":
    import sys
    print(json.dumps(LangfuseMCP().call(sys.argv[1], json.loads(sys.argv[2])), indent=2))
