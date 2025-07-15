from fastapi import APIRouter, HTTPException, Depends
from fastapi import Body
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from passlib.hash import bcrypt
from app.utils.task_helper import generate_daily_tasks_from_profile
from app.db.database import users_collection
from jose import jwt, JWTError
from fastapi.security import OAuth2PasswordBearer
import re
import dns.resolver
from fastapi import Form

from app.db.database import users_collection
SECRET_KEY="9Y5OC9hyv1UeOZtFa37AvR79IEURxN42MiDWZoRCLsE"
router = APIRouter()
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/vi/token")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/auth/token")


from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any

class ScheduleModel(BaseModel):
    breakfast_time: Optional[str] = Field(None, pattern=r"^\d{2}:\d{2}$")
    lunch_time: Optional[str] = Field(None, pattern=r"^\d{2}:\d{2}$")
    dinner_time: Optional[str] = Field(None, pattern=r"^\d{2}:\d{2}$")
    meditation_time: Optional[str] = Field(None, pattern=r"^\d{2}:\d{2}$")
    exercise_time: Optional[str] = Field(None, pattern=r"^\d{2}:\d{2}$")
    sleep_time: Optional[str] = Field(None, pattern=r"^\d{2}:\d{2}$")
    breathing_time: Optional[str] = Field(None, pattern=r"^\d{2}:\d{2}$")

class UserRegister(BaseModel):
    username: str
    password: str
    email: EmailStr
    date_of_birth: Optional[datetime] = None
    age: Optional[int] = Field(None, ge=0)
    gender: Optional[str] = Field(None, pattern="^(male|female|other)$")
    height_cm: Optional[int] = Field(None, ge=0)
    weight_kg: Optional[int] = Field(None, ge=0)
    profession: Optional[str] = None
    schedule: Optional[ScheduleModel] = None
    preferences: Optional[Dict[str, Any]] = None

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    age: Optional[int] = Field(None, ge=0)
    date_of_birth: Optional[datetime] = None
    gender: Optional[str] = Field(None, pattern="^(male|female|other)$")
    height_cm: Optional[int] = Field(None, ge=0)
    weight_kg: Optional[int] = Field(None, ge=0)
    profession: Optional[str] = None
    schedule: Optional[ScheduleModel] = None
    preferences: Optional[Dict[str, Any]] = None

class UserLogin(BaseModel):
    username: str
    password: str





def domain_exists(domain: str) -> bool:
    try:
        dns.resolver.resolve(domain, 'MX')
        return True
    except Exception:
        return False

# # oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
# class UserRegister(BaseModel):
#     username: str
#     password: str
#     email: EmailStr
#     age: Optional[int] = Field(default=None, ge=0)
#     # gender: Optional[str] = Field(default=None, pattern="")
#     gender: Optional[str] = Field(default=None, pattern="^(male|female|other)$")

# class UserLogin(BaseModel):
#     username: str
#     password: str

# class UserUpdate(BaseModel):
#     email: Optional[EmailStr]
#     age: Optional[int] = Field(default=None, ge=0)
#     # gender: Optional[str] = Field(default=None, regex="^(male|female|other)$")
#     gender: Optional[str] = Field(default=None, pattern="^(male|female|other)$")

def generate_token(username: str):
    from datetime import datetime, timedelta
    payload = {"sub": username, "exp": datetime.utcnow() + timedelta(days=1)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return True, payload["sub"]
    except JWTError as e:
        return False, str(e)

# @router.post("/register")
# def register(user: UserRegister):
#     if users_collection.find_one({"username": user.username}):
#         raise HTTPException(status_code=400, detail="Username already exists")
#     hashed = bcrypt.hash(user.password)
#     users_collection.insert_one({
#         "username": user.username,
#         "password": hashed,
#         "email": user.email,
#         "age": user.age,
#         "gender": user.gender,
#         "created_at": datetime.utcnow()
#     })
#     return {"message": "User registered successfully"}

@router.post("/register")
def register(user: UserRegister):
    

    def is_valid_email_format(email: str) -> bool:
        pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        return re.match(pattern, email) is not None

    if not is_valid_email_format(user.email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    # Extract domain and check if it exists
    domain = user.email.split("@")[-1].lower()
    if not domain_exists(domain):
        raise HTTPException(status_code=400, detail="Email domain does not exist")

    # Username or email already taken
    if users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed = bcrypt.hash(user.password)
    user_doc = {
        "username": user.username,
        "password": hashed,
        "email": user.email,
        "age": user.age,
        "date_of_birth": user.date_of_birth,
        "gender": user.gender,
        "height_cm": user.height_cm,
        "weight_kg": user.weight_kg,
        "profession": user.profession,
        "schedule": user.schedule.dict() if user.schedule else {},
        "preferences": user.preferences or {},
        "created_at": datetime.utcnow()
    }
    users_collection.insert_one(user_doc)

    # Optional: generate initial tasks
    generate_daily_tasks_from_profile(user_doc)

    return {"message": "User registered successfully"}


# @router.post("/login")
# def login(user: UserLogin):
#     record = users_collection.find_one({"username": user.username})
#     if not record or not bcrypt.verify(user.password, record["password"]):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     token = generate_token(user.username)
#     return {"access_token": token, "token_type": "bearer"}
@router.post("/token")
def login_for_access_token(username: str = Form(...), password: str = Form(...)):
    record = users_collection.find_one({"username": username})
    if not record or not bcrypt.verify(password, record["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = generate_token(username)
    return {"access_token": token, "token_type": "bearer"}


# @router.put("/update-profile")
# def update_profile(update: UserUpdate, token: str = Depends(oauth2_scheme)):
#     valid, username = decode_token(token)
#     if not valid:
#         raise HTTPException(status_code=401, detail="Invalid or expired token")
#     update_data = {k: v for k, v in update.dict().items() if v is not None}
#     if not update_data:
#         raise HTTPException(status_code=400, detail="No fields to update")
#     users_collection.update_one({"username": username}, {"$set": update_data})
#     return {"message": "Profile updated", "updated_fields": list(update_data.items())}

@router.put("/update-profile")
def update_profile(update: UserUpdate, token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    update_data = {
        k: (v.dict() if isinstance(v, BaseModel) else v)
        for k, v in update.dict().items() if v is not None
    }

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    users_collection.update_one({"username": username}, {"$set": update_data})
    return {"message": "Profile updated", "updated_fields": list(update_data.keys())}




@router.get("/me")
def get_profile(token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    user = users_collection.find_one({"username": username}, {"password": 0})

    if user and "_id" in user:
        user["_id"] = str(user["_id"])  # convert ObjectId to string

    return user


@router.post("/register-device-token")
def register_device_token(
    device_token: str = Body(...),
    token: str = Depends(oauth2_scheme)
):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail="Unauthorized")

    result = users_collection.update_one(
        {"username": username},
        {"$set": {"device_token": device_token}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    return {"message": "Device token saved"}