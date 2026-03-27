#!/usr/bin/env python3
"""Entry point for the Concrete FHE DAP debug server.

VS Code spawns this process and communicates via stdin/stdout using
the Debug Adapter Protocol (Content-Length framed JSON).
"""

import sys
import os

# Ensure the concrete-debugger package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from concrete_dap.server import DAPServer


def main():
    server = DAPServer()
    server.run()


if __name__ == "__main__":
    main()
