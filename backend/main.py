import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import MODEL_PATH, PREPROCESSOR_PATH
from .detector import PhishDetector
from .schemas import PredictRequest, PredictResponse

logger = logging.getLogger("phishguard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    detector = PhishDetector(MODEL_PATH, PREPROCESSOR_PATH)
    detector.load()
    app.state.detector = detector
    yield


app = FastAPI(
    title="PhishGuard AI API",
    description=(
        "URL-feature-only phishing detection using the trained XGBoost model. "
        "Confidence is the predicted class probability, not a guarantee of correctness."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://phishing-fake-website-detection-fro.vercel.app",
    ],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        errors.append({
            "field": field or "body",
            "message": error.get("msg", "Invalid value."),
        })

    return JSONResponse(
        status_code=422,
        content={"detail": errors},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "PhishGuard AI API is running"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> dict:
    detector = request.app.state.detector

    try:
        return detector.predict(payload.url)
    except Exception as exc:
        logger.exception("Prediction failed for a submitted URL")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        ) from exc
