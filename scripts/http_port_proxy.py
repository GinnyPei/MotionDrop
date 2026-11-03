#!/usr/bin/env python3
"""Small TCP proxy used to expose the loopback-only demo on a public port."""

from __future__ import annotations

import argparse
import select
import socket
import socketserver


class _ProxyHandler(socketserver.BaseRequestHandler):
    target_host: str
    target_port: int

    def handle(self) -> None:
        with socket.create_connection((self.target_host, self.target_port)) as upstream:
            peers = (self.request, upstream)
            while True:
                readable, _, _ = select.select(peers, (), ())
                for source in readable:
                    data = source.recv(64 * 1024)
                    if not data:
                        return
                    destination = upstream if source is self.request else self.request
                    destination.sendall(data)


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ProxyHandler.target_host = args.target_host
    _ProxyHandler.target_port = args.target_port
    with _ThreadingTCPServer((args.listen_host, args.listen_port), _ProxyHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
