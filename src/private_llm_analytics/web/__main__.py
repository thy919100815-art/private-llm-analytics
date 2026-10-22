"""CLI entrypoint for running the Private LLM Analytics Web Console (v0.5)."""

import argparse
import sys
import uvicorn

from private_llm_analytics.web.app import create_app


def main():
    parser = argparse.ArgumentParser(
        description="Private LLM Analytics Local Web Console"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1 - local only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)",
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Path to YAML configuration directory (default: configs)",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("Private LLM Analytics Console")
    print(f"http://{args.host}:{args.port}")
    print("Local trusted-administrator console. Raw telemetry remains local.")
    print("=" * 64)

    app = create_app(config_dir=args.config_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
