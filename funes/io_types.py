from langchain_core.pydantic_v1 import BaseModel, Field


class BasePaper(BaseModel):
    source_id: str = Field(description="the identifier of the paper in the source organization (e.g. arxiv)")
    title: str = Field(description="the title of the paper")
    url: str = Field(description="the url of the paper")