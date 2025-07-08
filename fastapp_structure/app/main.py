# from fastapi import FastAPI
# from app.api.v1.routes import router


# app = FastAPI()
# app.include_router(router, prefix='/api/v1')

from fastapi import FastAPI
from app.api.v1 import routes
from app.api.v2 import routes as v2_routes
# from app.api.v1.auth import router as auth_router
from app.api.auth import routes as auth_router


from app.db.database import db  # ✅ import the `db` object, not the module
from fastapi.middleware.cors import CORSMiddleware


from fastapi.openapi.utils import get_openapi

# ✅ Allow all origins, methods, and headers (adjust as needed for production)

from apscheduler.schedulers.background import BackgroundScheduler
from app.db.database import db
from app.utils.task_helper import generate_daily_tasks_from_profile,check_and_notify_pending_tasks_for_all_users
from app.utils.settings_secheduler import check_journal_times
app = FastAPI(title="Smart Assistant API")





scheduler = BackgroundScheduler()




def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version="1.0.0",
        description="Smart Assistant API",
        routes=app.routes,
    )

    # Fix tokenUrl in OpenAPI schema
    for scheme in openapi_schema.get("components", {}).get("securitySchemes", {}).values():
        if scheme.get("type") == "oauth2":
            flows = scheme.get("flows", {})
            if "password" in flows:
                flows["password"]["tokenUrl"] = "/auth/auth/token"

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


# Route registration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domain(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 🔐 Auth routes are version-free and global
app.include_router(auth_router.router, prefix="/auth", tags=["Authorization"])

app.include_router(routes.router, prefix="/api/v1")
app.include_router(v2_routes.router, prefix="/api/v2")




@app.get("/")
def read_root():
    return {"message": "Hello from Render"}


@app.get("/test-mongo")
def test_mongo():
    try:
        db.command("ping")  # ✅ valid on the actual DB object
        return {"message": "MongoDB connection successful!"}
    except Exception as e:
        return {"error": str(e)}





def scheduled_task_generation():
    print("⏰ Running daily task generation...")
    users = db["users"].find()
    for user in users:
        generate_daily_tasks_from_profile(user)

def scheduled_task_completion():
    print("⏰ Checking for pending tasks...")
    check_and_notify_pending_tasks_for_all_users()

def scheduled_journal_check():
    print("⏰ Running journal reminder check...")
    check_journal_times()



@app.on_event("startup")
def on_startup():

    # Register all jobs
    scheduler.add_job(scheduled_task_generation, 'cron', minute=30, id="generate_daily_tasks")
    scheduler.add_job(scheduled_task_completion, 'interval', seconds=30, id="check_pending_tasks")
    scheduler.add_job(scheduled_journal_check, 'interval', minutes=1, id="check_journal_times", misfire_grace_time=60)

    # Start scheduler
    scheduler.start()

