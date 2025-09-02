import argparse
import asyncio
import json
import os
import sys
from asyncio.subprocess import PIPE


class MCPStdIoClient:
    """Minimal MCP client speaking JSON-RPC 2.0 over stdio."""

    def __init__(self, proc):
        self.proc = proc
        self._next_id = 1

    def _next_request_id(self) -> int:
        req_id = self._next_id
        self._next_id += 1
        return req_id

    async def _send_request(self, method: str, params: dict | None = None) -> dict:
        if params is None:
            params = {}
        req = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": method,
            "params": params,
        }
        line = json.dumps(req) + "\n"
        self.proc.stdin.write(line.encode("utf-8"))
        await self.proc.stdin.drain()

        resp_line = await self.proc.stdout.readline()
        if not resp_line:
            raise RuntimeError("No response from MCP server")
        try:
            resp = json.loads(resp_line.decode("utf-8").strip())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from server: {e}: {resp_line!r}")
        if "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return resp["result"]

    async def initialize(self) -> dict:
        return await self._send_request("initialize", {"client": "mcp-client"})

    async def tools_list(self) -> dict:
        return await self._send_request("tools/list")

    async def tools_call(self, name: str, arguments: dict) -> dict:
        return await self._send_request("tools/call", {"name": name, "arguments": arguments})


async def spawn_mcp_server(project_root: str) -> asyncio.subprocess.Process:
    """Spawn the Python server in MCP-only mode and return the process."""
    server_path = os.path.join(project_root, "server", "mcp_server.py")
    if not os.path.exists(server_path):
        raise FileNotFoundError(f"Cannot find mcp_server.py at {server_path}")
    # Use python3, inherit current env
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        server_path,
        "--mcp-only",
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        cwd=os.path.dirname(server_path),
        env=os.environ.copy(),
    )
    return proc


async def run_client(args: argparse.Namespace) -> int:
    # Optionally spawn the server; otherwise, assume an existing stdio server is piped
    if args.spawn:
        proc = await spawn_mcp_server(args.project_root)
    else:
        raise SystemExit("This client currently requires --spawn to start the server process.")

    client = MCPStdIoClient(proc)

    try:
        if args.command == "init":
            result = await client.initialize()
            print(json.dumps(result, indent=2))
        elif args.command == "list-tools":
            # Ensure initialize first in case the server expects it
            await client.initialize()
            result = await client.tools_list()
            print(json.dumps(result, indent=2))
        elif args.command == "lookup-kb":
            await client.initialize()
            payload = {"query": args.query}
            if args.k is not None:
                payload["k"] = args.k
            result = await client.tools_call("lookup_kb", payload)
            print(json.dumps(result, indent=2))
        else:
            raise SystemExit(f"Unknown command: {args.command}")
        return 0
    finally:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        await proc.wait()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal MCP client for mcp_server.py")
    parser.add_argument("--project-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--spawn", action="store_true", help="Spawn the MCP server in --mcp-only mode")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Send initialize")
    sub.add_parser("list-tools", help="List available MCP tools")

    kb = sub.add_parser("lookup-kb", help="Call lookup_kb tool")
    kb.add_argument("--query", required=True, help="Question to ask")
    kb.add_argument("--k", type=int, default=None, help="Top documents to retrieve")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    exit_code = asyncio.run(run_client(args))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()



