from fastapi import FastAPI
from process_farmer_query import app as workflow_app
from pydantic import BaseModel, Field
from typing import Optional
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hi"}

@app.get("/hello")
def read_hello():
    return {"message": "Hello World"}

class FarmerQueryRequest(BaseModel):
    farmerId: str = Field(..., description="The unique identifier of the farmer")
    query: str = Field(..., description="The farmer's question or query")
    mediaUrl: Optional[str] = Field(None, description="Optional URL to an image or media file")

@app.post("/process-farmer-query")
async def process_farmer_query(request: FarmerQueryRequest):
    result = await workflow_app.ainvoke({
        "text": request.query,
        "imageUrl": request.mediaUrl,
        "entities": {},
        "service_response": {}
    })
    return {
        "farmerID": request.farmerId,
        "response": result.get("service_response", {})
    }
    
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8006))
    uvicorn.run(app, host="0.0.0.0", port=port)