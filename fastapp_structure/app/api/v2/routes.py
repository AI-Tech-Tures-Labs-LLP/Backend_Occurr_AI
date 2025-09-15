from fastapi import APIRouter
# from app.endpoints import user
# from .auth import router as auth_router
from app.api.v2 import chatbot, journal,notification,settings,health_data,adv_chat,op_chat,task_api
from app.api.v2 import breathing_exercise_api
from app.api.v2.dataconformation import router as data_conformation_router
from app.api.v2.dates import router as dates_router


router = APIRouter()
# router.include_router(user.router, prefix='/users', tags=['Users'])

# router.include_router(auth_router, prefix="/auth")
# router.include_router(chatbot.router, prefix="/chat",tags=["Chatbot (v2)"])
router.include_router(journal.router, prefix="/journal",tags=["Journal (v2)"])
router.include_router(settings.router, prefix="/schedule_journal_time",tags=["Schedule Journal Time (v2)"])
router.include_router(health_data.router, prefix="/health_data",tags=["Health Data (v2)"])
router.include_router(notification.router, prefix="/notification",tags=["Notification (v2)"])
router.include_router(adv_chat.router, prefix="/adv_chat",tags=["Advanced Chatbot (v2)"])
router.include_router(op_chat.router, prefix="/op_chat",tags=["Optimized Advanced Chatbot (v2)"])
router.include_router(breathing_exercise_api.router, prefix="/breathing_exercise", tags=["Breathing Exercise (v2)"])
router.include_router(task_api.router,prefix="/task",tags=["Tasks Operations"])
router.include_router(data_conformation_router, prefix="/data_conformation", tags=["Data Conformation"])
# router.include_router(vision_router, prefix="/vision", tags=["Vision Model"])
router.include_router(dates_router, prefix="/dates", tags=["User Dates"])