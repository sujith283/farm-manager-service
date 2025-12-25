#!/usr/bin/env python
"""
Run script for the farm manager service.
You can specify the port via environment variable PORT or command line argument.
"""
import os
import sys
import uvicorn
from main import app

if __name__ == "__main__":
    # Get port from command line argument or environment variable, default to 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Error: '{sys.argv[1]}' is not a valid port number")
            sys.exit(1)
    else:
        port = int(os.getenv("PORT", 8000))
    
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

