from __future__ import annotations
import uvicorn

if __name__ == "__main__":
    uvicorn.run("willrec.server.api:APP", host="0.0.0.0", port=8000, reload=False, workers=1)
