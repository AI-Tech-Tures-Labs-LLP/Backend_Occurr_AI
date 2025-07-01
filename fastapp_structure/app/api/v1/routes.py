from fastapi import APIRouter
# from app.endpoints import user
# from .auth import router as auth_router
from .chatbot import router as chat_router
from .journal import router as journal_router
from .settings import router as schedule_router
from .health_data import router as health_data_router
from  .notification import router as notification_router

router = APIRouter()
# router.include_router(user.router, prefix='/users', tags=['Users'])

# router.include_router(auth_router, prefix="/auth")
router.include_router(chat_router, prefix="/chat",tags=["Chatbot (v1)"])
router.include_router(journal_router,prefix="/journal",tags=["Journal (v1)"])
router.include_router(schedule_router,prefix="/schedule_journal_time",tags=["Schedule Journal Time (v1)"])
router.include_router(health_data_router,prefix="/health_data",tags=["Health Data (v1)"])
router.include_router(notification_router,prefix="/notification",tags=["Notification (v1)"])