from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from passlib.hash import bcrypt
from jose import jwt, JWTError
from fastapi.security import OAuth2PasswordBearer
from fastapi import Form

from app.db.database import users_collection
SECRET_KEY="9Y5OC9hyv1UeOZtFa37AvR79IEURxN42MiDWZoRCLsE"
router = APIRouter()
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/vi/token")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/token")

class UserRegister(BaseModel):
    username: str
    password: str
    email: EmailStr
    age: Optional[int] = Field(default=None, ge=0)
    # gender: Optional[str] = Field(default=None, pattern="")
    gender: Optional[str] = Field(default=None, pattern="^(male|female|other)$")

class UserLogin(BaseModel):
    username: str
    password: str

class UserUpdate(BaseModel):
    email: Optional[EmailStr]
    age: Optional[int] = Field(default=None, ge=0)
    # gender: Optional[str] = Field(default=None, regex="^(male|female|other)$")
    gender: Optional[str] = Field(default=None, pattern="^(male|female|other)$")

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

@router.post("/register")
def register(user: UserRegister):
    if users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed = bcrypt.hash(user.password)
    users_collection.insert_one({
        "username": user.username,
        "password": hashed,
        "email": user.email,
        "age": user.age,
        "gender": user.gender,
        "created_at": datetime.utcnow()
    })
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


@router.put("/update-profile")
def update_profile(update: UserUpdate, token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    update_data = {k: v for k, v in update.dict().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    users_collection.update_one({"username": username}, {"$set": update_data})
    return {"message": "Profile updated", "updated_fields": list(update_data.items())}


@router.get("/me")
def get_profile(token: str = Depends(oauth2_scheme)):
    valid, username = decode_token(token)
    if not valid:
        raise HTTPException(status_code=401, detail=username)

    user = users_collection.find_one({"username": username}, {"password": 0})

    if user and "_id" in user:
        user["_id"] = str(user["_id"])  # convert ObjectId to string

    return user