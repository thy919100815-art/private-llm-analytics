"""FastAPI application factory for the Private LLM Analytics Web Console (v0.5)."""

from pathlib import Path
from typing import Any, Optional
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from private_llm_analytics.privacy.exceptions import (
    PrivacyBudgetExceededError,
    PrivacyConfigurationError,
    ProductionRNGControlError,
    ProductionSafetyConfigurationError,
    RecipientSecurityViolation,
)
from private_llm_analytics.web.adapters import WebAnalyticsAdapter
from private_llm_analytics.web.schemas import (
    AccuracyEvaluationResponse,
    AdminPrivacyPolicyResponse,
    AdminPrivacyPolicyUpdateRequest,
    ConfigResponse,
    ContinualQueryRequest,
    ContinualStatusResponse,
    DatasetStatusResponse,
    DemoResetResponse,
    EpsilonSweepRequest,
    EpsilonSweepResponse,
    InputPreviewResponse,
    PlannerRecommendRequest,
    PlannerRecommendResponse,
    PrivacyStatusResponse,
    ReleaseRequest,
    ReleaseResponse,
)


def create_app(
    config_dir: str = "configs",
    adapter: Optional[WebAnalyticsAdapter] = None,
) -> FastAPI:
    """Create and configure local-only FastAPI application."""
    app = FastAPI(
        title="Private LLM Analytics Local Console",
        version="0.5.0-dev",
        docs_url="/docs",
        redoc_url=None,
    )

    web_adapter = adapter or WebAnalyticsAdapter(service_or_config_dir=config_dir)

    static_dir = Path(__file__).parent / "static"
    templates_dir = Path(__file__).parent / "templates"
    index_html_path = templates_dir / "index.html"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # -------------------------------------------------------------
    # Exception Handlers
    # -------------------------------------------------------------

    @app.exception_handler(PrivacyBudgetExceededError)
    async def budget_exceeded_handler(request: Any, exc: PrivacyBudgetExceededError):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": "Privacy budget exhausted. This release is not authorized under the current lifetime privacy cap.",
                "type": "PrivacyBudgetExceededError",
                "details": str(exc),
            },
        )

    @app.exception_handler(ProductionRNGControlError)
    async def rng_control_handler(request: Any, exc: ProductionRNGControlError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "External callers cannot supply RNG, seed, or noise parameters.",
                "type": "ProductionRNGControlError",
                "details": str(exc),
            },
        )

    @app.exception_handler(RecipientSecurityViolation)
    async def recipient_violation_handler(request: Any, exc: RecipientSecurityViolation):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": "Recipient access violation.",
                "type": "RecipientSecurityViolation",
                "details": str(exc),
            },
        )

    @app.exception_handler(PermissionError)
    async def permission_error_handler(request: Any, exc: PermissionError):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": str(exc),
                "type": "PermissionError",
                "details": "The recipient is not authorized for the requested metric or operation.",
            },
        )

    @app.exception_handler(ProductionSafetyConfigurationError)
    async def safety_config_handler(request: Any, exc: ProductionSafetyConfigurationError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Production safety configuration violation.",
                "type": "ProductionSafetyConfigurationError",
                "details": str(exc),
            },
        )

    @app.exception_handler(PrivacyConfigurationError)
    async def privacy_config_handler(request: Any, exc: PrivacyConfigurationError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Privacy configuration error.",
                "type": "PrivacyConfigurationError",
                "details": str(exc),
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Any, exc: ValueError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": str(exc),
                "type": "ValueError",
                "details": None,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Any, exc: RequestValidationError):
        first_error = exc.errors()[0] if exc.errors() else {"msg": "Validation error"}
        msg = str(first_error.get("msg", "Invalid request parameters"))
        loc = first_error.get("loc", ())
        field = loc[-1] if loc else "field"

        serialized_details = []
        for err in exc.errors():
            loc_str = ".".join(str(x) for x in err.get("loc", []))
            err_msg = str(err.get("msg", ""))
            serialized_details.append(f"{loc_str}: {err_msg}" if loc_str else err_msg)

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": f"Invalid input for '{field}': {msg}",
                "type": "RequestValidationError",
                "details": serialized_details,
            },
        )

    # -------------------------------------------------------------
    # HTML Dashboard Route
    # -------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def serve_index():
        if not index_html_path.exists():
            return HTMLResponse(
                content="<h1>Private LLM Analytics Console</h1><p>index.html not found</p>",
                status_code=404,
            )
        with open(index_html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)

    # -------------------------------------------------------------
    # API Routes
    # -------------------------------------------------------------

    @app.get("/api/health")
    async def health_check():
        return {
            "status": "ok",
            "version": "1.0.0",
            "service": "Private LLM Analytics Console",
            "local_only": True,
        }

    @app.get("/api/config", response_model=ConfigResponse)
    async def get_config():
        return web_adapter.get_config()

    @app.get("/api/dataset/status", response_model=DatasetStatusResponse)
    async def get_dataset_status():
        return web_adapter.get_dataset_status()

    @app.get("/api/dataset/sample-template")
    async def get_sample_template():
        content, filename = web_adapter.get_dataset_sample_template()
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/dataset/preview", response_model=InputPreviewResponse)
    async def get_dataset_preview(
        limit: int = Query(20, ge=1, le=100, description="Number of preview records"),
        recipient: Optional[str] = Query(None, description="Optional recipient parameter"),
    ):
        return web_adapter.get_dataset_preview(limit=limit, recipient=recipient)

    @app.post("/api/dataset/demo", response_model=DatasetStatusResponse)
    async def load_demo_data():
        return web_adapter.load_demo_data()

    @app.post("/api/dataset/upload", response_model=DatasetStatusResponse)
    async def upload_dataset(file: UploadFile = File(...)):
        content = await file.read()
        return web_adapter.load_uploaded_data(filename=file.filename or "telemetry.csv", content=content)

    # -------------------------------------------------------------
    # Demo-only Control Routes (Strictly Forbidden in Production)
    # -------------------------------------------------------------

    @app.post("/api/demo/reset-epoch", response_model=DemoResetResponse)
    async def reset_demo_epoch():
        if not web_adapter.is_demo_mode():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Demo reset endpoints are strictly forbidden in production mode.",
            )
        try:
            return web_adapter.reset_demo_epoch()
        except RecipientSecurityViolation as e:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @app.post("/api/demo/reset-all", response_model=DemoResetResponse)
    async def reset_demo_all():
        if not web_adapter.is_demo_mode():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Demo reset endpoints are strictly forbidden in production mode.",
            )
        try:
            return web_adapter.reset_demo_all()
        except RecipientSecurityViolation as e:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @app.get("/api/privacy/status", response_model=PrivacyStatusResponse)
    async def get_privacy_status(recipient: str = Query("provider_a")):
        return web_adapter.get_privacy_status(recipient=recipient)

    @app.post("/api/privacy/epoch/advance")
    async def advance_privacy_epoch():
        """Advance to a new 90-day Privacy Epoch."""
        if hasattr(web_adapter.service.orchestrator, "epoch_manager"):
            new_epoch = web_adapter.service.orchestrator.epoch_manager.advance_to_new_epoch()
            return new_epoch.to_dict()
        raise HTTPException(status_code=400, detail="Epoch manager not initialized.")

    @app.post("/api/planner/recommend", response_model=PlannerRecommendResponse)
    async def get_planner_recommendation(payload: PlannerRecommendRequest):
        return web_adapter.get_planner_recommendation(
            metric=payload.metric,
            recipient=payload.recipient,
            planning_horizon_days=payload.planning_horizon_days or 90,
        )

    @app.post("/api/release", response_model=ReleaseResponse)
    async def execute_release(payload: ReleaseRequest):
        return web_adapter.execute_release(payload)

    @app.get("/api/releases")
    async def list_releases():
        return web_adapter.list_releases()

    @app.get("/api/mechanism-cards")
    async def get_mechanism_cards():
        from private_llm_analytics.statistics.cards import get_all_mechanism_cards
        return get_all_mechanism_cards(web_adapter.service.config)

    @app.get("/api/admin/privacy-policy", response_model=AdminPrivacyPolicyResponse)
    async def get_admin_privacy_policy():
        return web_adapter.get_admin_policy()

    @app.post("/api/admin/privacy-policy", response_model=AdminPrivacyPolicyResponse)
    async def update_admin_privacy_policy(payload: AdminPrivacyPolicyUpdateRequest):
        return web_adapter.update_admin_policy(payload)

    @app.get("/api/release/download/csv")
    async def download_release_csv(
        release_id: str = Query(..., description="ID or canonical key of committed release"),
        recipient: str = Query("internal", description="Requesting recipient ID"),
    ):
        csv_content, filename = web_adapter.download_release_csv(release_id=release_id, recipient=recipient)
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/release/download/json")
    async def download_release_json(
        release_id: str = Query(..., description="ID or canonical key of committed release"),
        recipient: str = Query("internal", description="Requesting recipient ID"),
    ):
        json_content, filename = web_adapter.download_release_json(release_id=release_id, recipient=recipient)
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/release/download/internal-comparison")
    async def download_internal_comparison_csv(
        release_id: str = Query(..., description="ID or canonical key of committed release"),
        recipient: str = Query("internal", description="Requesting recipient ID (must be trusted)"),
    ):
        csv_content, filename = web_adapter.download_internal_comparison_csv(release_id=release_id, recipient=recipient)
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/evaluation/accuracy", response_model=AccuracyEvaluationResponse)
    async def get_release_accuracy(
        release_id: str = Query(..., description="ID or key of committed release"),
        recipient: str = Query("internal", description="Requesting recipient (must be trusted)"),
    ):
        return web_adapter.get_release_accuracy_evaluation(release_id=release_id, recipient=recipient)

    @app.get("/api/evaluation/summary")
    async def get_utility_summary(
        recipient: str = Query("internal", description="Requesting recipient (must be trusted)"),
    ):
        return web_adapter.get_utility_summary(recipient=recipient)

    @app.post("/api/evaluation/sweep", response_model=EpsilonSweepResponse)
    async def run_epsilon_sweep(payload: EpsilonSweepRequest):
        return web_adapter.run_epsilon_sweep(payload)

    @app.get("/api/evaluation/download/csv")
    async def download_evaluation_csv(
        release_id: str = Query(..., description="ID or key of committed release"),
        recipient: str = Query("internal", description="Requesting recipient (must be trusted)"),
    ):
        csv_content, filename = web_adapter.download_evaluation_csv(release_id=release_id, recipient=recipient)
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/evaluation/download/json")
    async def download_evaluation_json(
        release_id: str = Query(..., description="ID or key of committed release"),
        recipient: str = Query("internal", description="Requesting recipient (must be trusted)"),
    ):
        json_content, filename = web_adapter.download_evaluation_json(release_id=release_id, recipient=recipient)
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/continual/status", response_model=ContinualStatusResponse)
    async def get_continual_status():
        return web_adapter.get_continual_status()

    @app.post("/api/continual/query")
    async def query_continual(payload: ContinualQueryRequest):
        return web_adapter.query_continual_interval(payload)

    return app
