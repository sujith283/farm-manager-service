from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from process_farmer_query import app as workflow_app

router = APIRouter()

class FarmerQueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "farmerId": "farmer123",
                "query": "What are the government schemes for farmers in Andhra Pradesh?",
                "mediaUrl": ""
            }
        }
    )
    
    farmerId: str = Field(..., description="The unique identifier of the farmer", example="farmer123")
    query: str = Field(..., description="The farmer's question or query", example="What are the government schemes for mosambi farmers in Andhra Pradesh?")
    mediaUrl: str | None = Field(None, description="Optional URL to an image or media file", example="https://example.com/image.jpg")

class FarmerQueryResponse(BaseModel):
    farmerID: str = Field(..., description="The farmer ID from the request")
    response: dict = Field(..., description="The response from the service")

@router.post(
    "/process-farmer-query",
    response_model=FarmerQueryResponse,
    summary="Process Farmer Query",
    description="Process a farmer's query about schemes, crops, or both. The system will classify the intent and route to appropriate services.",
    response_description="Returns the processed response for the farmer's query"
)
async def process_farmer_query(request: FarmerQueryRequest):
    try:
        # Invoke the workflow (async nodes are handled automatically)
        result = await workflow_app.ainvoke({
            "text": request.query,
            "imageUrl": request.mediaUrl,
            "entities": {},
            "service_response": {}
        })
        
        # Extract response from workflow result
        response_data = result.get("service_response", {})
        
        return FarmerQueryResponse(
            farmerID=request.farmerId,
            response=response_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

