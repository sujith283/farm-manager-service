from fastapi import FastAPI
from service import router
import os

app = FastAPI()
app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Hi"}

@app.get("/hello")
def read_hello():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8006))
    uvicorn.run(app, host="0.0.0.0", port=port)

    
