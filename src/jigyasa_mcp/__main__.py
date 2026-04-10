"""Allow running as: python -m jigyasa_mcp"""

import sys
from jigyasa_mcp.cli import index_cli, mcp_cli

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mcp":
        sys.argv.pop(1)
        mcp_cli()
    else:
        index_cli()
