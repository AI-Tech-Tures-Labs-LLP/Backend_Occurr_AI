from fastapi import APIRouter
# from app.endpoints import user
# from .auth import router as auth_router
from app.api.v2 import chatbot, journal,notification,settings,health_data,adv_chat
router = APIRouter()
# router.include_router(user.router, prefix='/users', tags=['Users'])

# router.include_router(auth_router, prefix="/auth")
# router.include_router(chatbot.router, prefix="/chat",tags=["Chatbot (v2)"])
router.include_router(journal.router, prefix="/journal",tags=["Journal (v2)"])
router.include_router(settings.router, prefix="/schedule_journal_time",tags=["Schedule Journal Time (v2)"])
router.include_router(health_data.router, prefix="/health_data",tags=["Health Data (v2)"])
router.include_router(notification.router, prefix="/notification",tags=["Notification (v2)"])
router.include_router(adv_chat.router, prefix="/adv_chat",tags=["Advanced Chatbot (v2)"])